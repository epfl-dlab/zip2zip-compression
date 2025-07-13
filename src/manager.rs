use pyo3::prelude::*;

use crate::codec::{Codebook, CompressionState, decode_fn};
use crate::config::CompressionConfig;

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
        let batch_size = ids.len();

        if self.states.is_empty() {
            self.states = (0..batch_size)
                .map(|_| CompressionState::new(self.config.clone()))
                .collect();
            self.first_updates = true;
        }
        assert_eq!(batch_size, self.states.len());

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
