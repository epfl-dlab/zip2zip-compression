use pyo3::{exceptions, prelude::*, types::PyBytes};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::codec::{Codebook, CompressionState, decode_fn, encode_fn};
use crate::config::{CompressionConfig, PaddingStrategy, PaddingType};

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
