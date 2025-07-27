use itertools::Itertools;
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::compressor::LZWCompressor;
use crate::config::{CompressionConfig, PaddingStrategy};

pub const PADDING_INDEX: isize = -1;

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
#[pyclass(module = "zip2zip_compression")]
#[derive(Debug, Clone)]
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

    pub fn get_updates(&mut self, use_padding: bool) -> (Vec<usize>, Vec<isize>) {
        let size = if use_padding {
            self.config.max_codebook_size
        } else {
            self.updates.len()
        };

        let mut updates_vec: Vec<usize> =
            vec![self.config.pad_token_id; size * self.config.max_subtokens];
        let mut updates_indices: Vec<isize> = Vec::with_capacity(size);

        for (index, &id) in self.updates.iter().sorted().enumerate() {
            let start_index = index * self.config.max_subtokens;

            let entry = self.codebook.get_reverse(id).unwrap();
            updates_vec[start_index..start_index + entry.len()].copy_from_slice(entry);
            updates_indices.push(id as isize - self.config.initial_vocab_size as isize);
        }

        if use_padding {
            updates_vec.resize(size * self.config.max_subtokens, self.config.pad_token_id);
            updates_indices.resize(size, PADDING_INDEX);
        }

        self.updates.clear();
        (updates_vec, updates_indices)
    }
}

#[pymethods]
impl CompressionState {
    #[new]
    pub fn py_new(config: CompressionConfig) -> Self {
        Self::new(config)
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
