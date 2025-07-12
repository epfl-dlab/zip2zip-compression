use itertools::Itertools;
use pyo3::{exceptions, prelude::*, types::PyBytes};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// This is the config for the compression.
#[pyclass(get_all, module = "zip2zip_compression")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
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
impl CompressionConfig {
    #[new]
    pub fn new(
        initial_vocab_size: usize,
        max_codebook_size: usize,
        max_subtokens: usize,
        pad_token_id: usize,
        disabled_ids: Option<Vec<usize>>,
    ) -> Self {
        let mut disabled_ids =
            disabled_ids.map_or_else(|| HashSet::new(), |d_ids| d_ids.into_iter().collect());
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
#[pyclass(module = "zip2zip_compression")]
#[derive(Debug, Clone)]
pub struct Codebook {
    /// The actual compression (encoding) codebook.
    pub inner: HashMap<Vec<usize>, usize>,
    /// This is the de-compression (decoding) codebbok.
    pub reverse_inner: HashMap<usize, Vec<usize>>,
    /// The config of the compression.
    pub config: CompressionConfig,
}

impl Codebook {
    pub fn new(config: &CompressionConfig) -> Self {
        Self {
            inner: HashMap::with_capacity(config.max_codebook_size),
            reverse_inner: HashMap::with_capacity(config.max_codebook_size * config.max_subtokens),
            config: config.clone(),
        }
    }

    pub fn get(&self, ids: &Vec<usize>) -> Option<&usize> {
        self.inner.get(ids)
    }

    pub fn get_reverse(&self, id: usize) -> Option<&Vec<usize>> {
        self.reverse_inner.get(&id)
    }

    pub fn insert(&mut self, ids: Vec<usize>, id: usize) {
        let id = if id < self.config.initial_vocab_size {
            id + self.config.initial_vocab_size
        } else {
            id
        };

        self.inner.insert(ids.clone(), id);
        self.reverse_inner.insert(id, ids);
    }

    pub fn contains_key(&self, ids: &Vec<usize>) -> bool {
        self.inner.contains_key(ids)
    }
}

#[pymethods]
impl Codebook {
    /// Convert the codebook to a list of lists.
    pub fn to_list(&self, use_padding: bool) -> Vec<Vec<usize>> {
        let mut result = Vec::with_capacity(self.inner.len());

        for (ids, _) in self.inner.iter().sorted_by_key(|(_, id)| *id) {
            let mut entry: Vec<usize> = ids.clone();

            if use_padding {
                entry.resize(self.config.max_subtokens, self.config.pad_token_id);
            }

            result.push(entry);
        }

        if use_padding {
            result.resize(
                self.config.max_codebook_size,
                vec![self.config.pad_token_id; self.config.max_subtokens],
            );
        }

        result
    }

    /// Convert the codebook to a dictionary.
    ///
    /// The key is the hyper id and the value is the list of base ids that are merged to form the hyper id.
    ///
    /// The hyper id is the id of the hyper token.
    ///
    /// The base ids are the ids of the base tokens that are merged to form the hyper token.
    pub fn to_dict(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut result = BTreeMap::new();
        for (ids, id) in self.inner.iter() {
            result.insert(*id, ids.clone());
        }
        result
    }

    /// Get the subtokens for a given token id.
    ///
    /// If the token id is a base id, return None.
    ///
    /// If the token id is a hyper id, return the subtokens.
    ///
    /// The subtokens are the base ids that are merged to form the hyper id.
    pub fn get_subtokens(&self, id: usize) -> Option<Vec<usize>> {
        if id < self.config.initial_vocab_size {
            return None;
        }

        self.get_reverse(id).map(|x| x.clone())
    }
}

/// This struct contains the state of the compression. It is used to encode
/// and decode without keeping the state of the compressor.
pub struct CompressionState {
    /// The codebook of the compression.
    pub codebook: Codebook,
    /// The buffer of the compression. During the encoding, it is used to store
    /// the ids to merge. During the decoding, it is used to store the previous
    /// ids to merge.
    pub buffer: Vec<usize>,
    /// The next id to use for the compression.
    pub next_id: usize,
    /// The updates of the compression.
    pub updates: HashSet<usize>,
    /// The config of the compression.
    pub config: CompressionConfig,
}

impl CompressionState {
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            codebook: Codebook::new(&config),
            buffer: Vec::with_capacity(config.max_subtokens),
            next_id: config.initial_vocab_size,
            updates: HashSet::with_capacity(config.max_codebook_size),
            config,
        }
    }

    pub fn new_from_compressor(compressor: &LZWCompressor) -> Self {
        Self::new(compressor.config.clone())
    }

    pub fn get(&self, ids: &Vec<usize>) -> Option<&usize> {
        self.codebook.get(ids)
    }

    pub fn insert_buffer(&mut self, id: usize) {
        self.codebook.insert(self.buffer.clone(), id);
        self.updates.insert(id);
    }

    pub fn contains_key(&self, ids: &Vec<usize>) -> bool {
        self.codebook.contains_key(ids)
    }

    pub fn get_subtokens(&self, id: usize) -> Option<Vec<usize>> {
        if id < self.config.initial_vocab_size {
            return None;
        }

        self.codebook.get_reverse(id).map(|x| x.clone())
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

        for (index, &id) in self.updates.iter().sorted().enumerate() {
            let start_index = index * self.config.max_subtokens;

            let entry = self.codebook.get_reverse(id).unwrap();
            updates_vec[start_index..start_index + entry.len()].copy_from_slice(entry);
            updates_indices.push(id - self.config.initial_vocab_size);
        }

        if use_padding {
            updates_vec.resize(size * self.config.max_subtokens, self.config.pad_token_id);
        }

        self.updates.clear();
        (updates_vec, updates_indices)
    }
}

#[inline(always)]
fn codebook_contains(state: &CompressionState, ids: &Vec<usize>) -> bool {
    if ids.len() == 1 {
        ids[0] < state.config.initial_vocab_size
    } else {
        state.codebook.contains_key(ids)
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
pub fn encode_fn(
    state: &mut CompressionState,
    ids: &[usize],
    offset: usize,
    padding_strategy: PaddingStrategy,
    truncation: bool,
    max_length: Option<usize>,
) -> (Vec<usize>, usize) {
    let mut compressed_ids: Vec<usize> = Vec::new();

    let get_and_push = |compressed_ids: &mut Vec<usize>, state: &mut CompressionState| {
        if !truncation || compressed_ids.len() < max_length.unwrap() {
            let id = if state.buffer.len() == 1 {
                state.buffer[0]
            } else {
                *state.get(&state.buffer).unwrap()
            };
            compressed_ids.push(id);
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
        
        if state.config.disabled_ids.contains(&id) {
            if state.buffer.len() > 0 {
                get_and_push(&mut compressed_ids, state);
                state.buffer.clear();
            }
            state.buffer.push(id);
            get_and_push(&mut compressed_ids, state);
            state.buffer.clear();
            continue;
        }

        // check if the extended buffer is still a known code
        state.buffer.push(id);

        let is_in_codebook = codebook_contains(state, &state.buffer);
        //  if it's a brand new token, we can (1) emit the id for the buffer[:-1] (2) add the buffer to the codebook if still has space
        if !is_in_codebook {
            if state.next_id < state.config.initial_vocab_size + state.config.max_codebook_size {
                state.insert_buffer(state.next_id);
                state.next_id += 1;
            }

            state.buffer.pop();
            get_and_push(&mut compressed_ids, state);
            state.buffer.clear();
            state.buffer.push(id);
        }

        // reach the max number of subtokens, emit the buffer without adding new code
        if state.buffer.len() == state.config.max_subtokens {
            get_and_push(&mut compressed_ids, state);
            state.buffer.clear();
        }
    }

    // Handle the last buffer
    if !state.buffer.is_empty() {
        get_and_push(&mut compressed_ids, state);
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
                compressed_ids.resize(new_len, state.config.pad_token_id);
                // left padding
                compressed_ids.rotate_right(new_len - old_len);
            }
        }
        _ => {}
    }

    (compressed_ids, i)
}

/// Decode the compressed ids into a list of ids.
///
/// The `compressed_ids` is the list of compressed ids to decode.
///
/// Returns a list of ids.
pub fn decode_fn(state: &mut CompressionState, compressed_ids: &Vec<usize>) -> Vec<usize> {
    let mut output_ids: Vec<usize> = Vec::with_capacity(compressed_ids.len());

    for &id in compressed_ids {
        if state.config.disabled_ids.contains(&id) {
            state.buffer.clear();
            output_ids.push(id);
            continue;
        }

        let mut decoded_ids: Vec<usize>;

        // if the id is a base id, we can directly decode it
        if id < state.config.initial_vocab_size {
            decoded_ids = vec![id];
        // if the id is a known hyper id, we can decode it
        } else if let Some(base_ids) = state.get_subtokens(id) {
            decoded_ids = base_ids.clone();

        // now the id is not known, two cases:
        // 1. the id is a new hyper id because of force merge
        } else if state.buffer.len() == state.config.max_subtokens {
            decoded_ids = state.buffer.clone();
        // 2. the id is a new hyper id because of cScSc pattern merge
        } else {
            state.buffer.push(state.buffer[0]);

            state.insert_buffer(id);
            state.next_id += 1;
            output_ids.extend_from_slice(&state.buffer);
            continue;
        }
        // we have decoded the id, we can add it to the output
        output_ids.extend_from_slice(&decoded_ids);

        // the remaining part is to update the codebook if needed

        if state.next_id == state.config.initial_vocab_size + state.config.max_codebook_size {
            state.buffer.clear();
            continue;
        }

        // starting case
        if state.buffer.len() == 0 {
            state.buffer = decoded_ids;
            continue;
        }

        // the buffer is max size and the buffer is the previous ID
        // so it must not be a new hyper id but an existing one
        // we just clear the buffer and continue
        if state.buffer.len() == state.config.max_subtokens {
            state.buffer = decoded_ids.clone();
            continue;
        } else {
            while decoded_ids.len() > 0 {
                state.buffer.push(decoded_ids[0]);

                if !state.contains_key(&state.buffer) {
                    state.insert_buffer(state.next_id);
                    state.next_id += 1;
                    state.buffer = decoded_ids.clone();
                    break;
                } else if state.buffer.len() == state.config.max_subtokens {
                    // previous_ids = decoded_ids without the first element, could be empty
                    state.buffer = decoded_ids[1..].to_vec();
                    break;
                }

                decoded_ids.remove(0);
            }
        }
    }

    output_ids
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

#[pyclass(module = "zip2zip_compression")]
#[derive(Clone)]
pub struct LZWCompressor {
    pub config: CompressionConfig,
}

impl LZWCompressor {
    pub fn encode(
        &self,
        ids: &Vec<usize>,
        offset: usize,
        padding_strategy: PaddingStrategy,
        truncation: bool,
        max_length: Option<usize>,
    ) -> (Vec<usize>, CompressionState) {
        let mut state = CompressionState::new_from_compressor(self);
        let (compressed_ids, _) = encode_fn(
            &mut state,
            ids,
            offset,
            padding_strategy,
            truncation,
            max_length,
        );
        (compressed_ids, state)
    }

    pub fn decode(&self, compressed_ids: &Vec<usize>) -> (Vec<usize>, CompressionState) {
        let mut state = CompressionState::new_from_compressor(self);
        (decode_fn(&mut state, compressed_ids), state)
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

        match padding.unwrap() {
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
        Self {
            config: CompressionConfig::new(
                initial_vocab_size,
                max_codebook_size,
                max_subtokens,
                pad_token_id,
                disabled_ids,
            ),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.config).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle LZWCompressor: {e}"
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                self.config = serde_json::from_slice(s).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle LZWCompressor: {e}"
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
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
    #[pyo3(name = "encode", signature = (ids, padding = None, truncation = None, max_length = None))]
    pub fn py_encode(
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
        let (compressed_ids, state) =
            self.encode(&ids, 0, padding_strategy, truncation, max_length);

        let attention_mask = self.get_attention_mask(&compressed_ids);

        (
            compressed_ids,
            attention_mask,
            Py::new(py, state.codebook).unwrap(),
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
    #[pyo3(name = "decode")]
    pub fn py_decode(
        &self,
        py: Python<'_>,
        compressed_ids: Vec<usize>,
    ) -> (Vec<usize>, Py<Codebook>) {
        let (output_ids, state) = self.decode(&compressed_ids);

        (output_ids, Py::new(py, state.codebook).unwrap())
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
    #[pyo3(name = "batch_encode", signature = (ids, padding = None, truncation = None, max_length = None))]
    pub fn py_batch_encode(
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
        let (mut compressed_ids, codebooks): (Vec<Vec<usize>>, Vec<Codebook>) = ids
            .par_iter()
            .map(|ids| {
                let (compressed_ids, state) =
                    self.encode(ids, 0, padding_strategy, truncation, max_length);
                (compressed_ids, state.codebook)
            })
            .unzip();

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
    #[pyo3(name = "batch_decode")]
    pub fn py_batch_decode(
        &self,
        py: Python<'_>,
        compressed_ids: Vec<Vec<usize>>,
    ) -> Vec<(Vec<usize>, Py<Codebook>)> {
        let (compressed_ids, codebooks): (Vec<Vec<usize>>, Vec<Codebook>) = compressed_ids
            .par_iter()
            .map(|ids| {
                let (output_ids, state) = self.decode(ids);
                (output_ids, state.codebook)
            })
            .unzip();

        compressed_ids
            .into_iter()
            .zip(
                codebooks
                    .into_iter()
                    .map(|codebook| Py::new(py, codebook).unwrap()),
            )
            .collect()
    }

    /// Encode a sequence with a max_length constraint. This is used when pre-processing
    /// the dataset to avoid wasting samples. It will try to encode the sequence in chunks
    /// of `min_length` and `max_length` and return the compressed ids and the codebooks.
    ///
    /// The `ids` is the list of ids to encode.
    ///
    /// The `max_length` is the maximum length of the sequence.
    ///
    /// The `min_length` is the minimum length of the sequence.
    ///
    /// The `use_padding` is a boolean to indicate if the sequence should be padded.
    ///
    /// Returns a tuple containing the compressed ids and the codebooks.
    #[pyo3(name = "continuous_encode", signature = (ids, max_length, min_length = None, use_padding = None))]
    pub fn py_continuous_batch_encode(
        &self,
        ids: Vec<Vec<usize>>,
        max_length: usize,
        min_length: Option<usize>,
        use_padding: Option<bool>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<Vec<usize>>>) {
        let min_length = min_length.unwrap_or(0);
        let use_padding = use_padding.unwrap_or(true);
        let padding_strategy = if use_padding {
            PaddingStrategy::MaxLength
        } else {
            PaddingStrategy::DoNotPad
        };

        let (compressed_ids, codebooks): (Vec<Vec<usize>>, Vec<Codebook>) = ids
            .par_iter()
            .flat_map(|ids| {
                let mut offset = 0;
                let mut chunks = Vec::new();
                while min_length < (ids.len() - offset) {
                    let mut state = CompressionState::new_from_compressor(self);
                    let (c_ids, new_offset) = encode_fn(
                        &mut state,
                        &ids,
                        offset,
                        padding_strategy,
                        true,
                        Some(max_length),
                    );

                    offset = new_offset;
                    chunks.push((c_ids, state.codebook));
                }
                chunks
            })
            .unzip();

        let codebooks_as_lists = codebooks
            .par_iter()
            .map(|codebook| codebook.to_list(use_padding))
            .collect();

        (compressed_ids, codebooks_as_lists)
    }
}

/// The codebbok manager is a struct used to continue the creation of the codebook
/// during the generation. The codebook is initialized when compressing (encoding)
/// the input, but the model should be able to use hyper-tokens from the generation.
#[pyclass(module = "zip2zip_compression")]
pub struct CodebookManager {
    /// The states of the elements in the batch.
    pub states: Vec<CompressionState>,
    /// The first updates flag.
    pub first_updates: bool,
    /// The config of the codebook.
    pub config: CompressionConfig,
}

#[pymethods]
impl CodebookManager {
    #[new]
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            states: Vec::new(),
            first_updates: false,
            config,
        }
    }

    /// Get the subtokens for a single element in the batch.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// The `id` is the id to get the subtokens for.
    ///
    /// The `batch_index` is the index of the element in the batch.
    pub fn get_subtokens(&self, id: usize, batch_index: usize) -> Vec<usize> {
        self.states[batch_index]
            .get_subtokens(id)
            .unwrap_or_else(|| vec![id])
    }

    /// Update the codebooks for a single element in the batch.
    ///
    /// The `py` is the marker holding the GIL.
    ///
    /// The `ids` is the list of ids to update the codebooks with.
    ///
    /// Returns a tuple containing the updates and the indices of the updates.
    pub fn update_codebooks(&mut self, ids: Vec<Vec<usize>>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        // init the codebook for the first time, this depends on the batch, so it
        // is done in the first update.
        if self.states.is_empty() {
            self.states = ids
                .iter()
                .map(|_| CompressionState::new(self.config.clone()))
                .collect();
            self.first_updates = true;
        }
        assert_eq!(ids.len(), self.states.len());

        let (mut updates, updates_indices): (Vec<Vec<usize>>, Vec<Vec<usize>>) = self
            .states
            .iter_mut()
            .zip(ids.iter())
            .map(|(state, ids)| {
                if ids.len() == 1 && ids[0] == self.config.pad_token_id {
                    return (vec![], vec![]);
                }

                let _ = decode_fn(state, ids);

                // collect buffered updates from this state's codebook
                state.get_updates(self.first_updates) // still tell it whether this was the first call
            })
            .unzip();

        // If the sequence is only one token (not the first one), we need to
        // pad the updates to the longest sequence in the batch.
        if !self.first_updates {
            let max_length = updates.iter().map(|update| update.len()).max().unwrap();
            updates.iter_mut().for_each(|ids| {
                ids.resize(max_length, self.config.pad_token_id);
            });
        }
        self.first_updates = false;

        (updates, updates_indices)
    }

    pub fn get_codebooks(&self, py: Python<'_>) -> Vec<Py<Codebook>> {
        self.states
            .iter()
            .map(|state| Py::new(py, state.codebook.clone()).unwrap())
            .collect()
    }

    /// Reset the manager.
    ///
    /// This method is used to reset the manager when the generation is done.
    pub fn reset(&mut self) {
        self.states.clear();
    }
}

#[pymodule]
fn zip2zip_compression(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompressionConfig>()?;
    m.add_class::<Codebook>()?;
    m.add_class::<LZWCompressor>()?;
    m.add_class::<CodebookManager>()?;

    Ok(())
}
