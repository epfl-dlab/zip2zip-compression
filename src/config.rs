use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

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
