import torch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from huggingface_hub import hf_hub_download
from zip2zip_compression import LZWCompressor, CodebookConfig, CodebookManager

import pytest


VERBOSE = (
    False  # Set to True for detailed output, but will be ignored in multiprocessing
)

COMPRESSION_CONFIG_TYPE = "long"

BUILTIN_COMPRESSION_CONFIG_DICT = {
    "short": {
        "initial_vocab_size": 50257,
        "max_codebook_size": 2048,
        "max_subtokens": 4,
        "pad_token_id": 50256,
        "disabled_ids": [
            0,
            1,
            2,
            3,
            50256,
        ],  # 50256 must be in disabled_ids to distinguish pad token
    },
    "long": {
        "initial_vocab_size": 50257,
        "max_codebook_size": 16000,
        "max_subtokens": 16,
        "pad_token_id": 50256,
        "disabled_ids": [
            0,
            1,
            2,
            3,
            50256,
        ],  # 50256 must be in disabled_ids to distinguish pad token
    },
}

TARGET_ALGO = "fault_tolerant_lzw"


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

    # replace pad token with 1
    tokens = tokens.to(torch.int64)
    tokens[tokens == 50256] = 1

    if num_chunks is None:
        return tokens.view(-1, sequence_length)
    else:
        return tokens.view(-1, sequence_length)[:num_chunks]


def create_builtin_compressor(algo="fault_tolerant_lzw"):
    """Create and configure the LZW compressor."""
    if COMPRESSION_CONFIG_TYPE in BUILTIN_COMPRESSION_CONFIG_DICT:
        config = BUILTIN_COMPRESSION_CONFIG_DICT[COMPRESSION_CONFIG_TYPE]
        return LZWCompressor(
            initial_vocab_size=config["initial_vocab_size"],
            max_codebook_size=config["max_codebook_size"],
            max_subtokens=config["max_subtokens"],
            pad_token_id=config["pad_token_id"],
            disabled_ids=config["disabled_ids"],
        )
    else:
        raise ValueError(f"Invalid compressor type: {COMPRESSION_CONFIG_TYPE}")


def create_builtin_codebook_manager(algo="fault_tolerant_lzw"):
    if COMPRESSION_CONFIG_TYPE in BUILTIN_COMPRESSION_CONFIG_DICT:
        config = BUILTIN_COMPRESSION_CONFIG_DICT[COMPRESSION_CONFIG_TYPE]
        config = CodebookConfig(
            initial_vocab_size=config["initial_vocab_size"],
            max_codebook_size=config["max_codebook_size"],
            max_subtokens=config["max_subtokens"],
            pad_token_id=config["pad_token_id"],
            disabled_ids=set(config["disabled_ids"]),
        )
    else:
        raise ValueError(f"Invalid codebook manager type: {COMPRESSION_CONFIG_TYPE}")
    return CodebookManager(config, algorithm=algo)


def check_idempotence(token_chunk, lzw_compressor=None, verbose=False):
    """Process a single chunk for the idempotence test"""

    if lzw_compressor is None:
        lzw_compressor = create_builtin_compressor(algo=TARGET_ALGO)
    original_sequence = token_chunk.tolist()
    compressed_sequence, _, _ = lzw_compressor.encode(original_sequence)
    reconstructed_sequence, _ = lzw_compressor.decode(compressed_sequence)

    if verbose:
        _pairwise_colorprint(original_sequence, reconstructed_sequence)

    for i, (a, b) in enumerate(zip(original_sequence, reconstructed_sequence)):
        if a != b:
            return False, i, a, b
    return True, None, None, None


def check_decoding_corrupted_compression(
    token_chunk, lzw_compressor=None, verbose=False
):
    """Process a single chunk for the corrupted generation test"""
    if lzw_compressor is None:
        lzw_compressor = create_builtin_compressor(algo=TARGET_ALGO)
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


def check_codebook_consistency_online_vs_offline_decode(
    token_chunk, lzw_compressor=None, verbose=False
):
    """Process a single chunk for the codebook consistency test between decode and manager"""
    if lzw_compressor is None:
        lzw_compressor = create_builtin_compressor(algo=TARGET_ALGO)

    original_sequence = token_chunk.tolist()

    # Encode the data
    compressed_sequence, _, encode_codebook = lzw_compressor.encode(original_sequence)

    # Decode using lzw_compressor.decode to get the codebook
    decoded_sequence, decode_codebook = lzw_compressor.decode(compressed_sequence)

    # Verify the decoded sequence matches the original
    if decoded_sequence != original_sequence:
        for i, (a, b) in enumerate(zip(decoded_sequence, original_sequence)):
            if a != b:
                return False, i, a, b, "Decoded sequence doesn't match original"

    # Create a CodebookManager with the same configuration
    manager = create_builtin_codebook_manager()

    # Update the manager with the compressed sequence
    for i in range(len(compressed_sequence)):
        manager.update_codebooks([[compressed_sequence[i]]])

    # Get the codebooks from the manager
    manager_codebooks = manager.get_codebooks()
    if len(manager_codebooks) == 0:
        return False, 0, None, None, "No codebooks in manager"

    manager_codebook = manager_codebooks[0]

    # Compare the codebooks using the to_dict() method
    decode_dict = decode_codebook.to_dict()
    manager_dict = manager_codebook.to_dict()

    if verbose:
        print(f"Decode codebook size: {len(decode_dict)}")
        print(f"Manager codebook size: {len(manager_dict)}")
        print(f"Decode codebook: {decode_dict}")
        print(f"Manager codebook: {manager_dict}")

    # The codebooks should be identical
    if decode_dict != manager_dict:
        # Find the first difference
        all_keys = set(decode_dict.keys()) | set(manager_dict.keys())
        for key in sorted(all_keys):
            if key not in decode_dict:
                return False, key, None, None, f"Key {key} missing in decode codebook"
            if key not in manager_dict:
                return False, key, None, None, f"Key {key} missing in manager codebook"
            if decode_dict[key] != manager_dict[key]:
                return (
                    False,
                    key,
                    decode_dict[key],
                    manager_dict[key],
                    f"Values differ for key {key}",
                )

    return True, None, None, None, None


def check_codebook_consistency_during_online_generation(
    token_chunk, lzw_compressor=None, verbose=False
):
    """Process a single chunk for the codebook consistency test during online generation"""
    if lzw_compressor is None:
        lzw_compressor = create_builtin_compressor(algo=TARGET_ALGO)

    manager = create_builtin_codebook_manager(algo=TARGET_ALGO)

    seq_len = 10000
    prompt_len = 500
    assert prompt_len < seq_len

    original_sequence = token_chunk.tolist()

    # Encode the data
    compressed_total_sequence, _, total_codebook = lzw_compressor.encode(
        original_sequence
    )
    compressed_sequence, _, encode_codebook = lzw_compressor.encode(
        lzw_compressor.decode(compressed_total_sequence[:prompt_len])[0]
    )

    # first update
    manager.update_codebooks([compressed_sequence])

    # Corrupt the rest of the sequence
    for t in corrupt(compressed_total_sequence[prompt_len:], total_codebook):
        manager.update_codebooks([[t]])
    state = manager.states[0]

    for i, (c, c1) in enumerate(
        zip(
            total_codebook.to_list(use_padding=False),
            state.codebook.to_list(use_padding=False),
        )
    ):
        if c != c1:
            return False, i, c, c1, f"Codebook mismatch at index {i}: {c} != {c1}"

    return True, None, None, None, None


@pytest.fixture(scope="module")
def dataset():
    """Fixture to load the dataset once for all tests."""
    return load_dataset(num_chunks=100)


# @pytest.fixture(scope="module")
def lzw_compressor():
    """Fixture to create the compressor once for all tests."""
    return create_builtin_compressor()


def test_encode_decode_idempotence(
    dataset, *, lzw_compressor=None, num_processes=None
):
    """Test that encoding and then decoding preserves the original token sequence."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        process_fn = partial(check_idempotence, lzw_compressor=lzw_compressor)
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


def _process_test_decoding_corrupted_compression(args):
    """Helper function to process a chunk with its index for multiprocessing."""
    idx, chunk = args
    # Don't pass compressor — create it in the process
    return idx, check_decoding_corrupted_compression(chunk, verbose=VERBOSE)


def test_decoding_corrupted_compression(
    dataset, *, lzw_compressor=None, num_processes=None
):
    """
    Test decoding of a sequence that is a concatenation of encoded tokens and a 'corrupted' generation.
    The corrupted generation uses non-canonical codebook entries for some tokens.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    indexed_chunks = list(enumerate(dataset))  # [(0, chunk0), (1, chunk1), ...]
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(_process_test_decoding_corrupted_compression, indexed_chunks),
                total=len(indexed_chunks),
            )
        )

    for chunk_idx, (success, index, a, b) in results:
        assert (
            success
        ), f"Decoded content mismatch in chunk {chunk_idx} at index {index}: {a} != {b}"


def test_codebook_consistency_between_offline_and_online_decode(
    dataset, *, lzw_compressor=None, num_processes=None
):
    """Test that codebooks from lzw_compressor.decode and CodebookManager.update_codebooks are identical using real-world corpus."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        process_fn = partial(
            check_codebook_consistency_online_vs_offline_decode,
            lzw_compressor=lzw_compressor,
        )
        results = list(tqdm(pool.imap(process_fn, dataset), total=len(dataset)))

    for i, (success, index, a, b, error_msg) in enumerate(results):
        assert (
            success
        ), f"Codebook consistency check failed in chunk {i} at index {index}: {error_msg} (a={a}, b={b})"


def test_codebook_consistency_during_online_generation(
    dataset, *, lzw_compressor=None, num_processes=None
):
    """Test that codebooks from lzw_compressor.decode and CodebookManager.update_codebooks are identical during online generation."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        process_fn = partial(
            check_codebook_consistency_during_online_generation,
            lzw_compressor=lzw_compressor,
        )
        results = list(tqdm(pool.imap(process_fn, dataset), total=len(dataset)))

    for i, (success, index, a, b, error_msg) in enumerate(results):
        assert (
            success
        ), f"Codebook consistency during online generation failed in chunk {i} at index {index}: {error_msg} (a={a}, b={b})"


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