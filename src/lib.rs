pub mod codec;
pub mod compressor;
pub mod config;
pub mod manager;

use pyo3::prelude::*;

#[pymodule]
fn zip2zip_compression(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<config::CompressionConfig>()?;
    m.add_class::<codec::Codebook>()?;
    m.add_class::<compressor::LZWCompressor>()?;
    m.add_class::<manager::CodebookManager>()?;

    Ok(())
}
