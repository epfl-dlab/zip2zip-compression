# zip2zip-compression


## Installation



### From GitHub Release (no Rust required)

Find the matching wheel for your Python version and platform from the [GitHub Releases](https://github.com/epfl-dlab/zip2zip-compression/releases/tag/v0.2.0), then install directly using `pip`:

- `version` = `0.2.0`
- `python_version` = `312` for Python 3.12 (e.g., `310` for Python 3.10)
- `platform` = e.g., `manylinux_2_34_x86_64`, `win_amd64`, `macosx_11_0_arm64`

```bash
pip install https://github.com/epfl-dlab/zip2zip-compression/releases/download/v{version}/zip2zip_compression-{version}-cp{python_version}-cp{python_version}-{platform}.whl
# example:
pip install https://github.com/epfl-dlab/zip2zip-compression/releases/download/v0.2.0/zip2zip_compression-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl # for python 3.12 on x86_64
```



### From source(Rust required)

Make sure you have the Rust toolchain installed.


```bash
pip install maturin
maturin develop
```


## Example Usage

You can find usage examples in the [example](./example) folder.

Note: This package is primarily designed to be used as part of the [zip2zip](https://github.com/epfl-dlab/zip2zip) project, where it provides efficient, high-performance compression capabilities.
