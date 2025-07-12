# zip2zip-compression

> **zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression** \
> Saibo Geng, Nathan Ranchin, Yunzhen Yao, Maxime Peyrard, Chris Wendler, Michael Gastpar, Robert West \
> Paper: https://arxiv.org/abs/2506.01084

## About

This package provides a high-performance LZW compression library with Python bindings. It is designed to be used as part of the [zip2zip](https://github.com/epfl-dlab/zip2zip) project, where it provides efficient, high-performance compression capabilities.

We developed a new variant of the Lempel-Ziv-Welch (LZW) compression algorithm that doesn't need perfectly encoded input to decode. This allows the algorithm to decode (or decompress) generated sequences from a Large Language Model (LLM) without the need to store the entire compression codebook.

## Installation

### From PyPI

```bash
pip install zip2zip-compression
```

### From source (Rust required)

Make sure you have the Rust toolchain installed.

```bash
pip install maturin
maturin build --release
```

## Documentation

See the [docs](./docs) folder for more information:
- [Algorithm](./docs/algorithm.md)
- [Debug](./docs/debug.md)

## Example Usage

You can find usage examples in the [example](./example) folder.
