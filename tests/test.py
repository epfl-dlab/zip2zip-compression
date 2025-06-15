import pytest
from zip2zip_compression import LZWCompressor, CodebookConfig
from zip2zip_compression import CodebookManager as RuntimeCodebookManager
from zip2zip_compression import CodebookConfig


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
        disabled_ids=[26],
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
    decoded_ids = lzw.decode(compressed_ids, codebook)
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
        disabled_ids=[26],
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
    decoded_ids: list[list[int]] = lzw.batch_decode(compressed_ids, codebooks)
    unpadded_decoded_ids = [
        [id for id in ids if id != PAD_TOKEN_ID] for ids in decoded_ids
    ]
    if reversible:
        assert unpadded_decoded_ids == ids, "Decoded IDs mismatch"


def test_codebook_manager():


    config = CodebookConfig(
        initial_vocab_size=27,
        max_codebook_size=100,
        max_subtokens=5,
        pad_token_id=0,
        disabled_ids={26},
    )

    runtime_manager = RuntimeCodebookManager(config, algorithm="fault_tolerant_lzw")

    ids = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]

    updates, updates_indices = runtime_manager.update_codebooks(ids)

    print(f"\n--- {updates=} {updates_indices=}")
