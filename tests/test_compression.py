import pytest
from zip2zip_compression import LZWCompressor, CompressionConfig
from zip2zip_compression import CodebookManager as RuntimeCodebookManager

from zip2zip_compression import Codebook


# 4 Modes: Padding (X-axis) × Truncation (Y-axis)
#
#               Padding (X-axis)
#             No           Yes
#         ┌──────────┬──────────┐
#         │ Mode A   │ Mode B   │   No (variable length of inputs)
#         │          │          │
# Trunc.  ├──────────┼──────────┤
#   Yes   │ Mode C   │ Mode D   │   Yes (to a target length < max input length)
#         │          │          │
#         └──────────┴──────────┘
#
# Note: "Truncation" here effectively means using a fixed/static length,
# so a better name might be "static_length" vs "dynamic_length".

# ────────────────────────────────────────────────────────────────
# Mode A:
#   - No padding
#   - No truncation
#   → Output length = varies with input
#   → NOT aligned
#
# Mode B:
#   - Padding to longest input
#   - No truncation
#   → Output length = max(input lengths)
#   → Aligned
#
# Mode C:
#   - Truncate to target length
#   - No padding
#   → Output length = ≤ target length
#   → NOT aligned
#
# Mode D:
#   - Truncate to target length
#   - Pad to target length
#   → Output length = exactly target length
#   → Aligned


@pytest.mark.parametrize(
    "padding, truncation, max_length, expected_length",
    [
        ("do_not_pad", False, None, None),  # Mode A
        ("longest", False, None, 8),  # Mode B
        ("do_not_pad", True, 6, 6),  # Mode C
        ("max_length", True, 20, 20),  # Mode D
    ],
)
def test_encode_modes(padding, truncation, max_length, expected_length):
    PAD_TOKEN_ID = 0
    lzw = LZWCompressor(
        initial_vocab_size=27,
        max_codebook_size=100,
        max_subtokens=5,
        pad_token_id=PAD_TOKEN_ID,
        disabled_ids=[26, PAD_TOKEN_ID],
    )

    ids = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]  # len = 10

    compressed_ids, attention_mask, codebook = lzw.encode(
        ids,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )

    print(f"\n--- {padding=} {truncation=} {max_length=}")
    print("compressed_ids:", compressed_ids)
    print("attention_mask:", attention_mask)

    assert compressed_ids, "Compressed IDs should not be empty"

    if expected_length is not None:
        assert len(compressed_ids) == expected_length, "Length mismatch"
        assert len(attention_mask) == expected_length, "Attention mask mismatch"

    # test decode
    decoded_ids, _codebook = lzw.decode(compressed_ids)
    unpadded_decoded_ids = [id for id in decoded_ids if id != PAD_TOKEN_ID]
    assert (
        unpadded_decoded_ids == ids[: len(unpadded_decoded_ids)]
    ), "Decoded IDs mismatch"


@pytest.mark.parametrize(
    "padding, truncation, max_length, expected_lengths, reversible",
    [
        ("do_not_pad", False, None, (8, 5), True),  # Mode A
        ("longest", False, None, (8, 8), True),  # Mode B
        (
            "do_not_pad",
            True,
            6,
            (6, 5),
            False,
        ),  # Mode C (not reversible as we truncate the ids, so we get a shorter sequence  )
        ("max_length", True, 20, (20, 20), True),  # Mode D
    ],
)
def test_batch_encode_modes(
    padding, truncation, max_length, expected_lengths, reversible
):
    PAD_TOKEN_ID = 0
    ids = [
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ]

    lzw = LZWCompressor(
        initial_vocab_size=27,
        max_codebook_size=100,
        max_subtokens=5,
        pad_token_id=PAD_TOKEN_ID,
        disabled_ids=[26, PAD_TOKEN_ID],
    )

    compressed_ids, attention_masks, codebooks = lzw.batch_encode(
        ids, padding=padding, truncation=truncation, max_length=max_length
    )

    print(f"\n--- {padding=} {truncation=} {max_length=}")
    print("compressed_ids:", compressed_ids)
    print("attention_masks:", attention_masks)

    if expected_lengths is not None:
        assert (
            len(compressed_ids[0]) == expected_lengths[0]
        ), "Compressed IDs at index 0 length mismatch"
        assert (
            len(compressed_ids[1]) == expected_lengths[1]
        ), "Compressed IDs at index 1 length mismatch"

    # test batch decode
    decoded_ids_codebooks: list[tuple[list[int], Codebook]] = lzw.batch_decode(
        compressed_ids
    )
    decoded_ids = [ids for ids, _codebook in decoded_ids_codebooks]
    unpadded_decoded_ids = [
        [id for id in ids if id != PAD_TOKEN_ID] for ids in decoded_ids
    ]
    if reversible:
        assert unpadded_decoded_ids == ids, "Decoded IDs mismatch"


def test_codebook_manager():

    config = CompressionConfig(
        initial_vocab_size=27,
        max_codebook_size=100,
        max_subtokens=5,
        pad_token_id=0,
        disabled_ids=[26],
    )

    runtime_manager = RuntimeCodebookManager(config)

    ids = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]

    updates, updates_indices = runtime_manager.update_codebooks(ids)

    print(f"\n--- {updates=} {updates_indices=}")


def test_codebook_consistency_between_decode_and_manager():
    """Test that codebooks from lzw_compressor.decode and CodebookManager.update_codebooks are identical."""

    PAD_TOKEN_ID = 0
    config = CompressionConfig(
        initial_vocab_size=27,
        max_codebook_size=100,
        max_subtokens=5,
        pad_token_id=PAD_TOKEN_ID,
        disabled_ids=[26, PAD_TOKEN_ID],
    )

    lzw_compressor = LZWCompressor(
        initial_vocab_size=27,
        max_codebook_size=100,
        max_subtokens=5,
        pad_token_id=PAD_TOKEN_ID,
        disabled_ids=[26, PAD_TOKEN_ID],
    )

    # Create test data
    ids = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    # Encode the data
    compressed_ids, attention_mask, encode_codebook = lzw_compressor.encode(
        ids,
        padding="do_not_pad",
        truncation=False,
        max_length=None,
    )

    # Decode using lzw_compressor.decode to get the codebook
    decoded_ids, decode_codebook = lzw_compressor.decode(compressed_ids)

    # Verify the decoded ids match the original
    assert decoded_ids == ids, "Decoded IDs should match original IDs"

    # Now create a CodebookManager and update it with the same compressed ids
    manager = RuntimeCodebookManager(config)

    # Update the manager with the compressed ids
    for i in range(len(compressed_ids)):
        manager.update_codebooks([[compressed_ids[i]]])

    # Get the codebooks from the manager
    manager_codebooks = manager.get_codebooks()

    manager_codebook = manager_codebooks[0]

    # Compare the codebooks using the to_dict() method
    decode_dict = decode_codebook.to_dict()
    manager_dict = manager_codebook.to_dict()

    print(f"Decode codebook: {decode_dict}")
    print(f"Manager codebook: {manager_dict}")

    # The codebooks should be identical
    assert (
        decode_dict == manager_dict
    ), "Codebooks from decode and manager should be identical"

    print("✓ Codebooks are identical between decode and manager")
