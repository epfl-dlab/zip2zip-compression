use pyo3::prelude::*;

use crate::codec::{CompressionState, PADDING_INDEX, decode_fn};
use crate::config::CompressionConfig;

/// The codebbok manager is a struct used to continue the creation of the codebook
/// during the generation. The codebook is initialized when compressing (encoding)
/// the input, but the model should be able to use hyper-tokens from the generation.
#[pyclass(module = "zip2zip_compression")]
pub struct CodebookManager {
    /// The config of the codebook.
    pub config: CompressionConfig,
}

impl CodebookManager {
    /// Update the codebooks for a single element in the batch.
    ///
    /// The `ids` is the list of ids to update the codebooks with.
    ///
    /// The `states` is the list of states to update the codebooks with.
    pub fn update_codebooks(
        &mut self,
        ids: Vec<Vec<usize>>,
        states: &mut [&mut CompressionState],
    ) {
        assert_eq!(ids.len(), states.len());

        states.iter_mut().zip(ids.into_iter()).for_each(|(state, ids)| {
            if ids.len() == 1 && ids[0] == self.config.pad_token_id {
                return;
            }

            let _ = decode_fn(state, &ids);
        });
    }

    /// Get the updates for a single element in the batch.
    ///
    /// The `states` is the list of states to get the updates from.
    ///
    /// The `use_padding` is a boolean indicating whether to use padding.
    ///
    /// Returns a tuple containing the updates and the indices of the updates.
    pub fn get_updates(
        &self,
        states: &mut [&mut CompressionState],
        use_padding: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<isize>>) {
        let (mut updates, mut updates_indices): (Vec<Vec<usize>>, Vec<Vec<isize>>) = states
            .into_iter()
            .map(|state| {
                state.get_updates(use_padding)
            })
            .unzip();

        if !use_padding {
            let max_length = updates_indices
                .iter()
                .map(|indices| indices.len())
                .max()
                .unwrap_or(0);

            updates.iter_mut().for_each(|ids| {
                ids.resize(
                    max_length * self.config.max_subtokens,
                    self.config.pad_token_id,
                );
            });

            updates_indices.iter_mut().for_each(|indices| {
                indices.resize(max_length, PADDING_INDEX);
            });
        }

        (updates, updates_indices)
    }

    pub fn update_codebooks_and_get_updates(
        &mut self,
        ids: Vec<Vec<usize>>,
        mut states: Vec<&mut CompressionState>,
        use_padding: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<isize>>) {
        self.update_codebooks(ids, &mut states);
        self.get_updates(&mut states, use_padding)
    }
}

#[pymethods]
impl CodebookManager {
    #[new]
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    #[pyo3(name = "update_codebooks")]
    pub fn py_update_codebooks(
        &mut self,
        ids: Vec<Vec<usize>>,
        mut states: Vec<Py<CompressionState>>,
        py: Python<'_>,
    ) {
        let mut borrowed_states = states
            .iter_mut()
            .map(|state| state.borrow_mut(py))
            .collect::<Vec<_>>();

        let mut states_refs: Vec<&mut CompressionState> = borrowed_states
            .iter_mut()
            .map(|state| &mut **state)
            .collect();

        self.update_codebooks(ids, &mut states_refs)
    }

    #[pyo3(name = "get_updates")]
    pub fn py_get_updates(
        &self,
        mut states: Vec<Py<CompressionState>>,
        use_padding: bool,
        py: Python<'_>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<isize>>) {
        let mut borrowed_states = states
            .iter_mut()
            .map(|state| state.borrow_mut(py))
            .collect::<Vec<_>>();

        let mut states_refs: Vec<&mut CompressionState> = borrowed_states
            .iter_mut()
            .map(|state| &mut **state)
            .collect();

        self.get_updates(&mut states_refs, use_padding)
    }
}
