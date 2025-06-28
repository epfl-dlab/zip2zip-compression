use bumpalo::{collections::Vec as BumpVec, Bump};
use std::collections::{HashMap, HashSet, BTreeMap};
use itertools::Itertools;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// This is the config for the compression.
#[pyclass]
#[derive(Debug, Clone)]
pub struct CodebookConfig {
    /// The size of the vocabulary of the pre-trained tokenizer. This size
    /// includes also the added tokens.
    pub initial_vocab_size: usize,
    /// The maximum size of the LZW codebook.
    pub max_codebook_size: usize,
    /// The maxium number of normal tokens (non hyper-token) in a single
    /// codebook entry.
    pub max_subtokens: usize,
    /// The id of the padding token.
    pub pad_token_id: usize,
    /// The set of tokens id that cannot be marged with other tokens.
    /// For example, if `disable_ids = {42}` and we have a squence of tokens:
    /// [1, 42, 5, 6], we cannot create the hypertoken 7 = [1, 42] because
    /// 42 is disabled.
    pub disabled_ids: HashSet<usize>,
}

#[pymethods]
impl CodebookConfig {
    #[new]
    pub fn new(
        initial_vocab_size: usize,
        max_codebook_size: usize,
        max_subtokens: usize,
        pad_token_id: usize,
        disabled_ids: Option<HashSet<usize>>,
    ) -> Self {
        // insert the pad token id in the disabled ids
        let mut disabled_ids = disabled_ids.unwrap_or_else(|| HashSet::new());
        disabled_ids.insert(pad_token_id);

        Self {
            initial_vocab_size,
            max_codebook_size,
            max_subtokens,
            pad_token_id,
            disabled_ids,
        }
    }
}

/// This struct contains the state of the compression (encoding). This is
/// returned to the Python runtime to be used with the `CodebookManager`.
#[pyclass]
#[derive(Debug, Clone)]
pub struct Codebook {
    /// The actual compression (encoding) codebook.
    pub base_ids2hyper_id_map: HashMap<Vec<usize>, usize>,
    /// This is the de-compression (decoding) codebbok. This represents a
    /// HashMap<usize, Vec<usize>>, with all the entries padded to the
    /// `max_subtokens`.
    pub merges: Vec<usize>,
    /// This is the set of all the entries in the reverse codebook.
    pub active_hyper_ids: HashSet<usize>,
    /// This stored the updates to the codebook. This set is reset each
    /// time we call the `get_updates` method.
    updates: HashSet<usize>,
    /// This is stored after the compression (encoding) to continue the
    /// codebook creation.
    buffer_ids_to_merge: Vec<usize>,
    /// The compression config
    pub config: CodebookConfig,
}

impl Codebook {

    pub fn new(config: CodebookConfig) -> Self {
        Self {
            base_ids2hyper_id_map: HashMap::with_capacity(config.max_codebook_size),
            merges: vec![
                usize::MAX;
                config.max_codebook_size * config.max_subtokens
            ],
            active_hyper_ids: HashSet::with_capacity(config.max_codebook_size),
            updates: HashSet::with_capacity(config.max_codebook_size),
            buffer_ids_to_merge: Vec::with_capacity(config.max_subtokens),
            config,
        }
    }

    fn new_from_compressor(compressor: &LZWCompressor) -> Self {
        let config = compressor.config.clone();
        Self::new(config)
    }

    pub fn get(&self, base_ids: &Vec<usize>) -> Option<&usize> {
        self.base_ids2hyper_id_map.get(base_ids)
    }

    pub fn insert(&mut self, base_ids: Vec<usize>, hyper_id: usize) {
        // id is offset by the initial vocab size
        // in case not, offset the id by the initial vocab size
        let hyper_id = if hyper_id < self.config.initial_vocab_size {
            hyper_id + self.config.initial_vocab_size
        } else {
            hyper_id
        };
        self.base_ids2hyper_id_map.insert(base_ids.clone(), hyper_id);

        let index = hyper_id - self.config.initial_vocab_size;
        let start_index = index * self.config.max_subtokens;
        self.merges[start_index..start_index + base_ids.len()].copy_from_slice(&base_ids);
        self.active_hyper_ids.insert(hyper_id);
        self.updates.insert(hyper_id);
    }

    pub fn contains_key(&self, base_ids: &Vec<usize>) -> bool {
        self.base_ids2hyper_id_map.contains_key(base_ids)
    }

    pub fn get_updates(&mut self, use_padding: bool) -> (Vec<usize>, Vec<usize>) {
        let size = if use_padding {
            self.config.max_codebook_size
        } else {
            self.updates.len()
        };

        let mut updates_vec: Vec<usize> =
            vec![self.config.pad_token_id; size * self.config.max_subtokens];
        let mut updates_indices: Vec<usize> = Vec::with_capacity(size);

        for (update_index, &id) in self.updates.iter().sorted().enumerate() {
            let index = id - self.config.initial_vocab_size;
            let start_index = index * self.config.max_subtokens;

            let entry_length = self.merges[start_index..start_index + self.config.max_subtokens]
                .iter()
                .position(|&x| x == usize::MAX)
                .unwrap_or(self.config.max_subtokens);

            let end_index = start_index + entry_length;

            let target_range = update_index * self.config.max_subtokens..update_index * self.config.max_subtokens + entry_length;
            updates_vec[target_range.clone()]
                .copy_from_slice(&self.merges[start_index..end_index]);
            updates_indices.push(index);
        }

        if use_padding {
            updates_vec.resize(size * self.config.max_subtokens, self.config.pad_token_id);
        }

        self.updates.clear();
        (updates_vec, updates_indices)
    }

    pub fn get_base_ids(&self, hyper_id: usize) -> Option<Vec<usize>> {
        if hyper_id < self.config.initial_vocab_size {
            return None;
        }

        let index = hyper_id - self.config.initial_vocab_size;

        if self.active_hyper_ids.contains(&hyper_id) {
            let start_index = index * self.config.max_subtokens;
            let end_index = start_index + self.config.max_subtokens;
            let mut entry_vec = self.merges[start_index..end_index].to_vec();

            while entry_vec.last() == Some(&usize::MAX) {
                entry_vec.pop();
            }

            Some(entry_vec)
        } else {
            None
        }
    }
}

#[pymethods]
impl Codebook {
    /// Convert the codebook to a list of lists.
    ///
    /// If `use_padding` is true, the codebook will be padded to the
    /// `max_codebook_size` / `max_subtokens`.
    ///
    /// If `use_padding` is false, the codebook will be truncated to the
    /// `base_ids2hyper_id_map.len()`.
    pub fn to_list(&self, use_padding: bool) -> Vec<Vec<usize>> {
        let mut result = Vec::with_capacity(self.base_ids2hyper_id_map.len());
        let size = if use_padding {
            self.config.max_codebook_size / self.config.max_subtokens
        } else {
            self.base_ids2hyper_id_map.len()
        };

        for i in 0..size {
            let start_index = i * self.config.max_subtokens;
            let end_index = start_index + self.config.max_subtokens;
            let mut entry_vec: Vec<usize> =
                self.merges[start_index..end_index].to_vec();

            while entry_vec.last() == Some(&usize::MAX) {
                entry_vec.pop();
            }

            if use_padding {
                entry_vec.resize(self.config.max_subtokens, self.config.pad_token_id);
            }

            result.push(entry_vec);
        }
        result
    }

    pub fn to_dict(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut result = BTreeMap::new();
        for (ids, id) in self.base_ids2hyper_id_map.iter() {
            result.insert(*id, ids.clone());
        }
        result
    }
}

/// This enables the Union type in Python.
#[derive(FromPyObject)]
pub enum PaddingType {
    Str(String),
    Bool(bool),
}

/// The padding strategy. If the `PaddingType` is a boolean, if enabled,
/// the strategy is `PaddingStrategy::Longest`.
#[derive(Clone, Copy)]
pub enum PaddingStrategy {
    /// Pad the sequence to the longest sequence in the batch.
    Longest,
    /// Pad the sequence to the `max_length` specified.
    MaxLength,
    /// Do not pad the sequence.
    DoNotPad,
}

#[pyclass]
pub struct LZWCompressor {
    pub config: CodebookConfig,
}

impl LZWCompressor {
    #[inline(always)]
    fn codebook_contains(&self, codebook: &Codebook, ids: &Vec<usize>) -> bool {
        if ids.len() == 1 {
            ids[0] < self.config.initial_vocab_size
        } else {
            codebook.contains_key(ids)
        }
    }

    /// Encode the input ids into a compressed ids.
    ///
    /// The `offset` is the index of the first id to encode. This parameter is used
    /// to encode a very long sequence if we want to truncate it.
    ///
    /// The `padding_strategy` is the strategy to use to pad the sequence.
    ///
    /// The `truncation` is a boolean to indicate if the sequence should be truncated.
    ///
    /// The `max_length` is the maximum length of the sequence.
    ///
    /// Returns a tuple containing the compressed ids, the attention mask and the codebook.
    pub fn internal_encode(
        &self,
        ids: &[usize],
        offset: usize,
        padding_strategy: PaddingStrategy,
        truncation: bool,
        max_length: Option<usize>,
    ) -> ((Vec<usize>, Codebook), usize) {
        log::debug!("Hitting fn internal_encode");
        log::debug!("arg ids: {:?}", ids);
        log::debug!("arg offset: {}", offset);
        let mut compressed_ids: Vec<usize> = Vec::new();
        let mut codebook: Codebook = Codebook::new_from_compressor(self);

        let mut next_id: usize = self.config.initial_vocab_size;
        let mut buffer_ids_to_merge: Vec<usize> = Vec::with_capacity(self.config.max_subtokens);

        let get_and_push = |compressed_ids_ref: &mut Vec<usize>,
                            codebook_ref: &Codebook,
                            ids_to_push: &Vec<usize>| {
            if !truncation || compressed_ids_ref.len() < max_length.unwrap() {
                let id = if ids_to_push.len() == 1 {
                    ids_to_push[0]
                } else {
                    *codebook_ref.get(ids_to_push).unwrap()
                };
                compressed_ids_ref.push(id);
                log::debug!("emitting id: {}", id);
            }
        };

        let mut i = offset;
        while i < ids.len() {
            // check if we need to early exit because of truncation
            if truncation && max_length.is_some() && compressed_ids.len() >= max_length.unwrap() {
                break;
            }

            let id = ids[i];
            i += 1;
            log::debug!("coming id: {}", id);
            log::debug!("buffer_ids_to_merge: {:?}", buffer_ids_to_merge);

            if self.config.disabled_ids.contains(&id) {
                if buffer_ids_to_merge.len() > 0 {
                    get_and_push(&mut compressed_ids, &codebook, &buffer_ids_to_merge);
                    buffer_ids_to_merge.clear();
                    log::debug!("force emitting buffer_ids_to_merge because of disabled id: {}", id);
                }
                get_and_push(&mut compressed_ids, &codebook, &vec![id]);
                log::debug!("emitting disabled id: {}", id);
                continue;
            }

            // check if the extended buffer is still a known code
            buffer_ids_to_merge.push(id);

            let is_in_codebook = self.codebook_contains(&codebook, &buffer_ids_to_merge);
            //  if it's a brand new token, we can (1) emit the id for the buffer[:-1] (2) add the buffer to the codebook if still has space
            if !is_in_codebook {
                if next_id < self.config.initial_vocab_size + self.config.max_codebook_size {
                    codebook.insert(buffer_ids_to_merge.clone(), next_id);
                    log::debug!("inserting: {:?} -> {:?}", buffer_ids_to_merge, next_id);
                    next_id += 1;
                }

                buffer_ids_to_merge.pop();
                get_and_push(&mut compressed_ids, &codebook, &buffer_ids_to_merge);
                buffer_ids_to_merge.clear();
                buffer_ids_to_merge.push(id);
            }

            // reach the max number of subtokens, emit the buffer without adding new code
            if buffer_ids_to_merge.len() == self.config.max_subtokens {
                log::debug!("force emitting buffer_ids_to_merge because of max_subtokens reached: {:?}", buffer_ids_to_merge);
                get_and_push(&mut compressed_ids, &codebook, &buffer_ids_to_merge);
                buffer_ids_to_merge.clear();
            }
        }

        codebook.buffer_ids_to_merge = buffer_ids_to_merge.clone();

        // Handle the last buffer
        if !buffer_ids_to_merge.is_empty() {
            log::debug!("force emitting buffer_ids_to_merge because reaching the end of the ids");
            get_and_push(&mut compressed_ids, &codebook, &buffer_ids_to_merge);
        }

        match padding_strategy {
            PaddingStrategy::MaxLength => {
                // if the padding strategy is max_length, we need to pad the sequence to the max length
                // check if the max_length is provided
                if max_length.is_none() {
                    panic!("max_length is not provided");
                }
                let new_len = max_length.unwrap();
                if compressed_ids.len() < new_len {
                    let old_len = compressed_ids.len();
                    compressed_ids.resize(new_len, self.config.pad_token_id);
                    // left padding
                    compressed_ids.rotate_right(new_len - old_len);
                }
            }
            _ => {}
        }
        // Return (compressed sequence, codebook) and number of input tokens consumed
        // i is the number of input tokens consumed, it can be used for next iteration
        ((compressed_ids, codebook), i)
    }

    /// Decode the compressed ids into a list of ids.
    ///
    /// The `compressed_ids` is the list of compressed ids to decode.
    ///
    /// Returns a list of ids.
    pub fn internal_decode(&self, compressed_ids: &Vec<usize>) -> (Vec<usize>, Codebook) {
        log::debug!("Hitting fn internal_decode");
        log::debug!("arg compressed_ids: {:?}", compressed_ids);
        let bump = Bump::new();

        let mut output_ids: Vec<usize> = Vec::with_capacity(compressed_ids.len());
        let mut codebook: Codebook = Codebook::new_from_compressor(self);

        let mut next_id: usize = self.config.initial_vocab_size;
        let mut previous_ids: &[usize] = &[];

        for &id in compressed_ids {
            if self.config.disabled_ids.contains(&id) {
                previous_ids = &[];
                output_ids.push(id);
                log::debug!("Received disabled id: {}, clearing buffer_ids_to_merge", id);
                continue;
            }

            let decoded_ids: &[usize];

            // if the id is a base id, we can directly decode it
            if id < self.config.initial_vocab_size {
                decoded_ids = bump.alloc_slice_copy(&[id]);
            // if the id is a known hyper id, we can decode it
            } else if let Some(slice) = codebook.get_base_ids(id) {
                decoded_ids = bump.alloc_slice_copy(&slice);

            // now the id is not known, two cases:
            // 1. the id is a new hyper id because of force merge
            } else if previous_ids.len() == self.config.max_subtokens {
                decoded_ids = previous_ids;
            // 2. the id is a new hyper id because of cSc pattern merge
            } else {
                assert!(id == next_id, "id: {} != next_id: {}", id, next_id);
                let mut inferred_vec = BumpVec::with_capacity_in(previous_ids.len() + 1, &bump);
                inferred_vec.extend_from_slice(previous_ids);
                inferred_vec.push(previous_ids[0]);

                decoded_ids = inferred_vec.into_bump_slice();

                codebook.insert(decoded_ids.to_vec(), id);
                log::debug!("inserting: {:?} -> {:?}", decoded_ids, id);
                next_id += 1;
            }

            output_ids.extend_from_slice(decoded_ids);

            // Now we need to add a new entry to the codebook if the conditions are met:
            if !previous_ids.is_empty()
                && next_id < self.config.initial_vocab_size + self.config.max_codebook_size
                && previous_ids.len() < self.config.max_subtokens
            {
                let mut new_entry_vec = BumpVec::with_capacity_in(previous_ids.len() + 1, &bump);
                new_entry_vec.extend_from_slice(previous_ids);
                new_entry_vec.push(decoded_ids[0]);

                let new_entry_slice = new_entry_vec.into_bump_slice();

                codebook.insert(new_entry_slice.to_vec(), next_id);
                log::debug!("inserting: {:?} -> {:?}", new_entry_slice, next_id);
                next_id += 1;
            }

            // update the previous ids
            previous_ids = decoded_ids;
        }

        (output_ids, codebook)
    }

    pub fn internal_fuzzy_decode(&self, compressed_ids: &Vec<usize>) -> (Vec<usize>, Codebook) {
        log::debug!("Hitting fn internal_fuzzy_decode");
        log::debug!("arg compressed_ids: {:?}", compressed_ids);
        let mut output_ids: Vec<usize> = Vec::with_capacity(compressed_ids.len());
        let mut codebook: Codebook = Codebook::new_from_compressor(self);

        let mut next_id: usize = self.config.initial_vocab_size;
        let mut previous_ids: Vec<usize> = Vec::new();

        for &id in compressed_ids {
            log::debug!("coming id: {}", id);
            log::debug!("buffer_ids_to_merge: {:?}", previous_ids);
            if self.config.disabled_ids.contains(&id) {
                previous_ids.clear();
                output_ids.push(id);
                log::debug!("emitting disabled id: {}", id);
                continue;
            }

            let mut decoded_ids: Vec<usize>;

            // if the id is a base id, we can directly decode it
            if id < self.config.initial_vocab_size {
                decoded_ids = vec![id];
            // if the id is a known hyper id, we can decode it
            } else if let Some(base_ids) = codebook.get_base_ids(id) {
                decoded_ids = base_ids.clone();

            // now the id is not known, two cases:
            // 1. the id is a new hyper id because of force merge
            } else if previous_ids.len() == self.config.max_subtokens {
                decoded_ids = previous_ids.clone();
            // 2. the id is a new hyper id because of cScSc pattern merge
            } else {
                log::debug!("Unkown id: {}, because of cScSc pattern merge", id);
                decoded_ids = previous_ids.clone();
                decoded_ids.push(previous_ids[0]);

                codebook.insert(decoded_ids.clone(), id);
                log::debug!("inserting: {:?} -> {:?}", decoded_ids, id);
                next_id += 1;
                previous_ids = decoded_ids.clone();
                output_ids.extend_from_slice(&decoded_ids);
                log::debug!("emitting id: {:?}", decoded_ids);
                continue;
            }
            // we have decoded the id, we can add it to the output
            output_ids.extend_from_slice(&decoded_ids);
            log::debug!("emitting id: {:?}", decoded_ids);

            // the remaining part is to update the codebook if needed

            if next_id == self.config.initial_vocab_size + self.config.max_codebook_size {
                log::debug!("max codebook size reached, clearing buffer_ids_to_merge");
                previous_ids.clear();
                continue;
            }

            // starting case
            if previous_ids.len() == 0 {
                previous_ids = decoded_ids;
                continue;
            }

            // the buffer is max size and the buffer is the previous ID
            // so it must not be a new hyper id but an existing one
            // we just clear the buffer and continue
            if previous_ids.len() == self.config.max_subtokens {
                assert!(codebook.contains_key(&previous_ids), "previous_ids: {:?} not in codebook: {:?}", previous_ids, codebook);
                log::debug!("force emitting buffer_ids_to_merge because of max_subtokens reached: {:?}", previous_ids);
                previous_ids = decoded_ids.clone();
                continue;
            } else {

                while decoded_ids.len() > 0
                {
                    previous_ids.push(decoded_ids[0]);

                    if !codebook.contains_key(&previous_ids) {
                        codebook.insert(previous_ids.clone(), next_id);
                        log::debug!("inserting: {:?} -> {:?}", previous_ids, next_id);
                        next_id += 1;
                        previous_ids = decoded_ids.clone();
                        break;
                    } else if previous_ids.len() == self.config.max_subtokens {
                        // previous_ids = decoded_ids.clone();
                        log::debug!("force emitting buffer_ids_to_merge because of max_subtokens reached: {:?}", previous_ids);
                        // previous_ids = decoded_ids without the first element, could be empty
                        previous_ids = decoded_ids[1..].to_vec();
                        break;
                    }

                    decoded_ids.remove(0);
                }
            }
        }

        //print the codebook
        log::debug!("Final codebook built from fuzzy decode: {:?}", codebook.to_dict());

        (output_ids, codebook)
    }

    #[inline(always)]
    fn get_attention_mask(&self, compressed_ids: &Vec<usize>) -> Vec<usize> {
        compressed_ids
            .iter()
            .map(|&id| (id != self.config.pad_token_id) as usize)
            .collect()
    }

    /// Get the padding strategy from the `padding` parameter.
    ///
    /// The `padding` is a string or a boolean.
    ///
    /// If the `padding` is a string, it can be "longest" or "max_length".
    ///
    /// If the `padding` is a boolean, it can be true or false.
    ///
    /// Returns the padding strategy.
    #[inline(always)]
    fn get_padding_strategy(&self, padding: Option<PaddingType>) -> PaddingStrategy {
        if padding.is_none() {
            return PaddingStrategy::DoNotPad;
        }

        let padding = padding.unwrap();

        match padding {
            PaddingType::Str(padding_str) => {
                if padding_str == "longest" {
                    PaddingStrategy::Longest
                } else if padding_str == "max_length" {
                    PaddingStrategy::MaxLength
                } else {
                    PaddingStrategy::DoNotPad
                }
            }
            PaddingType::Bool(padding_bool) => {
                if padding_bool {
                    PaddingStrategy::Longest
                } else {
                    PaddingStrategy::DoNotPad
                }
            }
        }
    }
}

#[pymethods]
impl LZWCompressor {
    #[new]
    pub fn new(
        initial_vocab_size: usize,
        max_codebook_size: usize,
        max_subtokens: usize,
        pad_token_id: usize,
        disabled_ids: Option<Vec<usize>>,
    ) -> Self {
        let disabled_ids = disabled_ids.map_or_else(
            || HashSet::with_capacity(0),
            |d_ids| d_ids.into_iter().collect(),
        );

        Self {
            config: CodebookConfig::new(
                initial_vocab_size,
                max_codebook_size,
                max_subtokens,
                pad_token_id,
                Some(disabled_ids)
            ),
        }
    }

    /// Encode the input ids into a compressed ids.
    ///
    /// The `ids` is the list of ids to encode.
    ///
    /// The `padding` is the padding strategy to use.
    ///
    /// The `truncation` is a boolean to indicate if the sequence should be truncated.
    ///
    /// The `max_length` is the maximum length of the sequence.
    ///
    /// Returns a tuple containing the compressed ids, the attention mask and the codebook.
    pub fn encode(
        &self,
        py: Python<'_>,
        ids: Vec<usize>,
        padding: Option<PaddingType>,
        truncation: Option<bool>,
        max_length: Option<usize>,
    ) -> (Vec<usize>, Vec<usize>, Py<Codebook>) {
        let truncation = truncation.unwrap_or(false);
        assert!(!truncation || max_length.is_some());

        let padding_strategy = self.get_padding_strategy(padding);
        let ((compressed_ids, codebook), _) =
            self.internal_encode(&ids, 0, padding_strategy, truncation, max_length);

        let attention_mask = self.get_attention_mask(&compressed_ids);

        (
            compressed_ids,
            attention_mask,
            Py::new(py, codebook).unwrap(),
        )
    }

    /// Decode the compressed ids into a list of ids.
    ///
    /// The `compressed_ids` is the list of compressed ids to decode.
    ///
    /// The `codebook` is the codebook to use for decoding. If not provided, the codebook will be inferred from the
    /// compressed ids.
    ///
    /// Returns a list of ids.
    pub fn decode(
        &self,
        compressed_ids: Vec<usize>
    ) -> (Vec<usize>, Codebook) {
        self.internal_fuzzy_decode(&compressed_ids)
    }

    /// Encode a batch of input ids into a batch of compressed ids.
    ///
    /// The `ids` is the list of ids to encode.
    ///
    /// The `padding` is the padding strategy to use.
    ///
    /// The `truncation` is a boolean to indicate if the sequence should be truncated.
    ///
    /// The `max_length` is the maximum length of the sequence.
    ///
    /// Returns a tuple containing the compressed ids, the attention mask and the codebook.
    pub fn batch_encode(
        &self,
        py: Python<'_>,
        ids: Vec<Vec<usize>>,
        padding: Option<PaddingType>,
        truncation: Option<bool>,
        max_length: Option<usize>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Py<Codebook>>) {
        let truncation = truncation.unwrap_or(false);
        assert!(!truncation || max_length.is_some());

        let padding_strategy = self.get_padding_strategy(padding);
        let (outputs, _): (Vec<(Vec<usize>, Codebook)>, Vec<usize>) = ids
            .par_iter()
            .map(|ids| self.internal_encode(ids, 0, padding_strategy, truncation, max_length))
            .unzip();

        let (mut compressed_ids, codebooks): (Vec<Vec<usize>>, Vec<Codebook>) =
            outputs.into_iter().unzip();

        match padding_strategy {
            PaddingStrategy::Longest => {
                let max_length = compressed_ids.iter().map(|ids| ids.len()).max().unwrap();
                compressed_ids.iter_mut().for_each(|ids| {
                    let old_len = ids.len();
                    ids.resize(max_length, self.config.pad_token_id);
                    ids.rotate_right(max_length - old_len);
                });
            }
            _ => {}
        }

        let attention_masks = compressed_ids
            .iter()
            .map(|compressed_ids| self.get_attention_mask(compressed_ids))
            .collect();

        (
            compressed_ids,
            attention_masks,
            codebooks
                .into_iter()
                .map(|codebook| Py::new(py, codebook).unwrap())
                .collect(),
        )
    }

    /// Decode a batch of compressed ids into a batch of ids.
    ///
    /// The `compressed_ids` is the list of compressed ids to decode.
    ///
    /// The `codebooks` is the list of codebooks to use for decoding. If not provided, the codebooks will be inferred from the
    /// compressed ids.
    ///
    /// Returns a list of ids.
    pub fn batch_decode(
        &self,
        compressed_ids: Vec<Vec<usize>>
    ) -> Vec<(Vec<usize>, Codebook)> {
        compressed_ids
                .par_iter()
                // .map(|ids| self.internal_decode(ids))
                .map(|ids| self.internal_fuzzy_decode(ids))
                .collect()
    }
}

/// The state associated with a `Codebook`. This is used to enables multiple
/// configs in a same batch.
#[pyclass]
#[derive(Debug, Clone)]
pub struct CodebookState {
    /// The codebook to use as a reference to a Python object.
    codebook: Py<Codebook>,
    /// The ids to merge. This is used to create a new entry in the codebook.
    buffer_ids_to_merge: Vec<usize>,
    /// The next id to use.
    next_id: usize,
    /// The config of the codebook.
    config: CodebookConfig,
}

/// The codebbok manager is a struct used to continue the creation of the codebook
/// during the generation. The codebook is initialized when compressing (encoding)
/// the input, but the model should be able to use hyper-tokens from the generation.
#[pyclass]
pub struct CodebookManager {
    /// The algorithm to use for updating the codebook.
    #[pyo3(get)]
    algorithm: String,

    /// The states of the elements in the batch.
    #[pyo3(get)]
    states: Vec<CodebookState>,
    /// The first updates flag.
    first_updates: bool,
    /// The config of the codebook.
    #[pyo3(get)]
    config: CodebookConfig,
}

impl CodebookManager {
    #[inline(always)]
    fn codebook_contains(codebook: &Codebook, ids: &Vec<usize>) -> bool {
        if ids.len() == 1 {
            ids[0] < codebook.config.initial_vocab_size
        } else {
            codebook.contains_key(ids)
        }
    }

    /// Update the codebook for a single element in the batch.
    fn internal_update_codebook(py: Python<'_>, state: &mut CodebookState, ids: &[usize]) {
        let config = &state.config;

        // If the sequence is only one token and it is the pad token, we don't
        // update the codebook because it happens at the end of the generation
        // for one element in the batch while the other elements are still
        // generating.
        if ids.len() == 1 && ids[0] == config.pad_token_id {
            return;
        }

        let mut codebook = state.codebook.borrow_mut(py);
        for &maybe_hid in ids {
            let ids_to_process = codebook
                .get_base_ids(maybe_hid)
                .unwrap_or_else(|| vec![maybe_hid]);

            for id in ids_to_process {
                if config.disabled_ids.contains(&id) {
                    state.buffer_ids_to_merge.clear();
                    continue;
                }

                state.buffer_ids_to_merge.push(id);

                let is_in_codebook = Self::codebook_contains(&codebook, &state.buffer_ids_to_merge);

                if !is_in_codebook {
                    if state.next_id < config.initial_vocab_size + config.max_codebook_size {
                        codebook.insert(state.buffer_ids_to_merge.clone(), state.next_id);
                        state.next_id += 1;
                    }
                    state.buffer_ids_to_merge.clear();
                    state.buffer_ids_to_merge.push(id);
                }

                if state.buffer_ids_to_merge.len() == config.max_subtokens {
                    state.buffer_ids_to_merge.clear();
                }
            }
        }
    }


    fn internal_fuzzy_update_codebook(
        py: Python<'_>,
        state: &mut CodebookState,
        ids: &[usize],
    ) {
        log::debug!("Hitting fn internal_fuzzy_update_codebook");
        log::debug!("arg ids: {:?}", ids);

        let config = &state.config;
        // if ids.len() == 1 && ids[0] == config.pad_token_id {
        //     return;
        // }

        let mut codebook = state.codebook.borrow_mut(py);

        for &maybe_hid in ids {
            log::debug!("coming maybe_hid: {}", maybe_hid);
            log::debug!("buffer_ids_to_merge: {:?}", state.buffer_ids_to_merge);
            if config.disabled_ids.contains(&maybe_hid) {
                // no need to update the codebook, because buffer_ids_to_merge is already in the codebook
                // and buffer_ids_to_merge + maybe_hid is forbidden
                log::debug!("Got disabled id: {}, clearing buffer_ids_to_merge", maybe_hid);
                state.buffer_ids_to_merge.clear();
                continue;
            }

            let mut current_ids: Vec<usize>;
            if maybe_hid < config.initial_vocab_size {
                current_ids = vec![maybe_hid];
            } else if let Some(base_ids) = codebook.get_base_ids(maybe_hid) {
                current_ids = base_ids.clone();
            } else { // (2) cSc pattern
                log::debug!("Unknown id: {}, because of cSc pattern merge", maybe_hid);
                current_ids = state.buffer_ids_to_merge.clone();
                current_ids.push(state.buffer_ids_to_merge[0]);

                // check if maybe_hid is equal to next_id
                if maybe_hid != state.next_id {
                    panic!("maybe_hid != state.next_id");
                }

                codebook.insert(current_ids.clone(), maybe_hid);
                log::debug!("inserting: {:?} -> {:?}", current_ids, maybe_hid);
                state.next_id += 1;
                state.buffer_ids_to_merge = current_ids.clone();
                continue;

            }

            if state.next_id == config.initial_vocab_size + config.max_codebook_size {
                log::debug!("max_codebook_size reached, clearing buffer_ids_to_merge");
                state.buffer_ids_to_merge.clear();
                continue;
            }

            // Starting time
            if state.buffer_ids_to_merge.len() == 0 {
                state.buffer_ids_to_merge = current_ids.clone();
                continue;
            }

            if state.buffer_ids_to_merge.len() == config.max_subtokens {
                log::debug!("max_subtokens reached, clearing buffer_ids_to_merge");
                state.buffer_ids_to_merge = current_ids.clone();
                continue;
            } else {
                while  current_ids.len() > 0 {
                    state.buffer_ids_to_merge.push(current_ids[0]);

                    if !codebook.contains_key(&state.buffer_ids_to_merge) {
                        codebook.insert(state.buffer_ids_to_merge.clone(), state.next_id);
                        log::debug!("inserting: {:?} -> {:?}", state.buffer_ids_to_merge, state.next_id);
                        state.next_id += 1;
                        state.buffer_ids_to_merge = current_ids.clone();
                        break;
                    } else if state.buffer_ids_to_merge.len() == config.max_subtokens {
                        log::debug!("max_subtokens reached, clearing buffer_ids_to_merge");
                        // remove the first element
                        state.buffer_ids_to_merge = current_ids[1..].to_vec();
                        break;
                    }
                    current_ids.remove(0);
                }
            }
        }
    }
}

#[pymethods]
impl CodebookManager {
    #[new]
    pub fn new(config: CodebookConfig, algorithm: Option<&str>) -> Self {
        Self {
            algorithm: algorithm.unwrap_or("renormalizing_lzw").to_string(),
            states: Vec::new(),
            first_updates: false,
            config,
        }
    }

    /// Set the codebooks for the manager.
    ///
    /// The `codebooks` is the list of codebooks to set.
    ///
    /// The codebooks are set as a reference to a Python object.
    pub fn set_codebooks(&mut self, codebooks: Vec<&PyCell<Codebook>>) {
        let num_codebooks = codebooks.len();
        self.states.clear();
        self.states.reserve(num_codebooks);

        for codebook_cell in codebooks {
            let codebook = codebook_cell.borrow();
            let config = codebook.config.clone();

            self.states.push(CodebookState {
                codebook: Py::from(codebook_cell),
                buffer_ids_to_merge: codebook.buffer_ids_to_merge.clone(),
                next_id: config.initial_vocab_size + codebook.base_ids2hyper_id_map.len(),
                config,
            });
        }

        self.first_updates = true;
    }

    /// Get the subtokens for a single element in the batch.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// The `id` is the id to get the subtokens for.
    ///
    /// The `batch_index` is the index of the element in the batch.
    pub fn get_subtokens(&self, py: Python<'_>, id: usize, batch_index: usize) -> Vec<usize> {
        self.states[batch_index]
            .codebook
            .borrow(py)
            .get_base_ids(id)
            .unwrap_or_else(|| vec![id])
    }

    /// Update the codebooks for a single element in the batch.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// The `ids` is the list of ids to update the codebooks with.
    ///
    /// Returns a tuple containing the updates and the indices of the updates.
    pub fn update_codebooks(
        &mut self,
        py: Python<'_>,
        ids: Vec<Vec<usize>>,
    ) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        log::debug!("Hitting fn update_codebooks");
        log::debug!("arg ids: {:?}", ids);
        // init the codebook for the first time, this depends on the batch, so it
        // is done in the first update.
        if self.states.is_empty() && self.algorithm == "fault_tolerant_lzw" {
            let batch_size = ids.len();
            let mut codebooks = Vec::with_capacity(batch_size);

            for _ in 0..batch_size {
                let codebook = Py::new(py, Codebook::new(self.config.clone()))?;
                codebooks.push(codebook.into_ref(py));
            }

            self.set_codebooks(codebooks);
        }// renormalizing_lzw requires the codebook to be initialized with codebook from the tokenizer

        assert_eq!(ids.len(), self.states.len());
        let max_ids_length = ids.iter().map(|i| i.len()).max().unwrap();


        let (mut updates, updates_indices): (Vec<Vec<usize>>, Vec<Vec<usize>>) = self
        .states
        .iter_mut()
        .zip(ids.iter())
        .map(|(state, ids)| {
            // choose the implementation for every call, even the first one
            match self.algorithm.as_str() {
                "fault_tolerant_lzw"   => {
                    CodebookManager::internal_fuzzy_update_codebook(py, state, ids)
                }
                "renormalizing_lzw" => {
                    if !self.first_updates {
                        CodebookManager::internal_update_codebook(py, state, ids)
                    }
                }
                _ => panic!("Invalid algorithm: {}", self.algorithm),
            }

            // collect buffered updates from this state's codebook
            state
                .codebook
                .borrow_mut(py)
                .get_updates(self.first_updates)   // still tell it whether this was the first call
        })
        .unzip();

        // If the sequence is only one token (not the first one), we need to
        // pad the updates to the longest sequence in the batch.
        if !self.first_updates {
            let max_length = updates.iter().map(|update| update.len()).max().unwrap();
            updates
                .iter_mut()
                .zip(self.states.iter())
                .for_each(|(ids, state)| {
                    let pad_token_id = state.codebook.borrow(py).config.pad_token_id;
                    ids.resize(max_length, pad_token_id);
                });
        }
        self.first_updates = false;

        Ok((updates, updates_indices))
    }

    pub fn get_codebooks(&self) -> Vec<Py<Codebook>> {
        self.states
            .iter()
            .map(|st| st.codebook.clone())   // clone the Py<…> handle
            .collect()
    }

    /// Reset the manager.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// This method is used to reset the manager when the generation is done.
    pub fn reset(&mut self, py: Python<'_>) {
        // if the algorithm is "renormalizing_lzw", we need to keep the buffer_ids_to_merge
        if self.algorithm == "renormalizing_lzw" {
            for state in self.states.iter_mut() {
                let mut codebook = state.codebook.borrow_mut(py);
                codebook.buffer_ids_to_merge = state.buffer_ids_to_merge.clone();
            }
        }

        self.states.clear();
    }
}


#[pymodule]
fn zip2zip_compression(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    env_logger::init();
    m.add_class::<CodebookConfig>()?;
    m.add_class::<Codebook>()?;
    m.add_class::<CodebookState>()?;
    m.add_class::<LZWCompressor>()?;
    m.add_class::<CodebookManager>()?;

    Ok(())
}
