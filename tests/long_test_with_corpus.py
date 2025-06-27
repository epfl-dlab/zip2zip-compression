import torch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from huggingface_hub import hf_hub_download
from zip2zip_compression import LZWCompressor

import pytest


VERBOSE = (
    False  # Set to True for detailed output, but will be ignored in multiprocessing
)


def load_dataset(sequence_length=10_000, num_chunks=None):
    """Load and validate the dataset from HuggingFace."""
    filename = hf_hub_download(
        repo_id="kjj0/fineweb10B-gpt2",
        filename="fineweb_train_000001.bin",
        repo_type="dataset",
    )

    # Validate header
    header = torch.from_file(filename, False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])

    # Load tokens
    with open(filename, "rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"

    if num_chunks is None:
        return tokens.view(-1, sequence_length)
    else:
        return tokens.view(-1, sequence_length)[:num_chunks]


def create_compressor():
    """Create and configure the LZW compressor."""
    return LZWCompressor(
        initial_vocab_size=50257,
        max_codebook_size=2048,
        max_subtokens=4,
        pad_token_id=50256,
        disabled_ids=[
            0,
            1,
            2,
            3,
            50256,
        ],  # 50256 must be in disabled_ids to distinguish pad token
    )


def check_chunk_idempotence(token_chunk, lzw_compressor=None, verbose=False):
    """Process a single chunk for the idempotence test"""

    if lzw_compressor is None:
        lzw_compressor = create_compressor()
    original_sequence = token_chunk.tolist()
    compressed_sequence, _, _ = lzw_compressor.encode(original_sequence)
    reconstructed_sequence, _ = lzw_compressor.decode(compressed_sequence)

    if verbose:
        _pairwise_colorprint(original_sequence, reconstructed_sequence)

    for i, (a, b) in enumerate(zip(original_sequence, reconstructed_sequence)):
        if a != b:
            return False, i, a, b
    return True, None, None, None


def check_chunk_corrupted(token_chunk, lzw_compressor=None, verbose=False):
    """Process a single chunk for the corrupted generation test"""
    if lzw_compressor is None:
        lzw_compressor = create_compressor()
    original_tokens = token_chunk.tolist()
    encoded_tokens, _, encoding_codebook = lzw_compressor.encode(original_tokens)
    corrupted_generation = corrupt(encoded_tokens, encoding_codebook)
    decoded_tokens, decoding_codebook = lzw_compressor.decode(corrupted_generation)
    expected_sequence = original_tokens + original_tokens

    encoding_codebook_dict = encoding_codebook.to_dict()
    decoding_codebook_dict = decoding_codebook.to_dict()
    # check if the encoding and decoding codebooks are the identical
    assert (
        encoding_codebook_dict == decoding_codebook_dict
    ), "Encoding and decoding codebooks are not identical"

    if verbose:
        _pairwise_colorprint(decoded_tokens, expected_sequence)

    for i, (a, b) in enumerate(zip(decoded_tokens, expected_sequence)):
        if a != b:
            return False, i, a, b
    return True, None, None, None


@pytest.fixture(scope="module")
def dataset():
    """Fixture to load the dataset once for all tests."""
    return load_dataset()


# @pytest.fixture(scope="module")
def lzw_compressor():
    """Fixture to create the compressor once for all tests."""
    return create_compressor()


def test_encode_decode_idempotence(dataset, *, lzw_compressor=None, num_processes=None):
    """Test that encoding and then decoding preserves the original token sequence."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        process_fn = partial(check_chunk_idempotence, lzw_compressor=lzw_compressor)
        results = list(tqdm(pool.imap(process_fn, dataset), total=len(dataset)))

    for i, (success, index, a, b) in enumerate(results):
        assert (
            success
        ), f"Decoded content mismatch in chunk {i} at token {index}: {a} != {b}"


def corrupt(encoded_tokens, codebook):
    """
    Repeat the input tokens but 'corrupt' some by using non-canonical codebook entries.
    """
    output = []
    codebook_list = codebook.to_list(use_padding=False)
    is_canonical = torch.randint(
        0, 2, (len(encoded_tokens),), generator=torch.Generator().manual_seed(42)
    )
    for token_id, use_canonical in zip(encoded_tokens, is_canonical):
        if use_canonical == 0 and token_id >= 50257:
            subtokens = codebook_list[token_id - 50257]
            output.extend(subtokens)
        else:
            output.append(token_id)
    return output


def _process_chunk_with_index(args):
    """Helper function to process a chunk with its index for multiprocessing."""
    idx, chunk = args
    # Don't pass compressor — create it in the process
    return idx, check_chunk_corrupted(chunk, verbose=VERBOSE)


def test_corrupted_generation_decoding(
    token_chunks, *, lzw_compressor=None, num_processes=None
):
    """
    Test decoding of a sequence that is a concatenation of encoded tokens and a 'corrupted' generation.
    The corrupted generation uses non-canonical codebook entries for some tokens.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    indexed_chunks = list(enumerate(token_chunks))  # [(0, chunk0), (1, chunk1), ...]
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(_process_chunk_with_index, indexed_chunks),
                total=len(indexed_chunks),
            )
        )

    for chunk_idx, (success, index, a, b) in results:
        assert (
            success
        ), f"Decoded content mismatch in chunk {chunk_idx} at index {index}: {a} != {b}"


def _pairwise_colorprint(seq1, seq2):
    """Helper function to print pairs of elements from two sequences."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    for a, b in zip(seq1, seq2):
        equal = a == b
        sign = "==" if equal else "!="
        if equal:
            print(f"{GREEN}{a} {sign} {b}{RESET}")
        else:
            print(f"{RED}{a} {sign} {b}{RESET}")


if __name__ == "__main__":
    num_processes = mp.cpu_count()  # Use all available CPU cores
    # num_processes = 1  # For debugging, set to 1 to avoid multiprocessing issues

    print(f"Running tests with {num_processes} processes...")

    input_dataset = load_dataset()
    test_corrupted_generation_decoding(input_dataset, num_processes=num_processes)
    test_encode_decode_idempotence(input_dataset, num_processes=num_processes)
    print("All tests passed successfully.")
