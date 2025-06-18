import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from zip2zip_compression import LZWCompressor

import pytest


filename = hf_hub_download(
    repo_id="kjj0/fineweb10B-gpt2",
    filename="fineweb_train_000001.bin",
    repo_type="dataset",
)

header = torch.from_file(filename, False, 256, dtype=torch.int32)
assert header[0] == 20240520, "magic number mismatch in the data .bin file"
assert header[1] == 1, "unsupported version"
num_tokens = int(header[2])

with open(filename, "rb", buffering=0) as f:
    tokens = torch.empty(num_tokens, dtype=torch.uint16)
    f.seek(256 * 4)
    nbytes = f.readinto(tokens.numpy())
    assert nbytes == 2 * num_tokens, "number of tokens read does not match header"

compressor = LZWCompressor(
    initial_vocab_size=50257,
    max_codebook_size=2048,
    max_subtokens=8,
    pad_token_id=50256,
    disabled_ids=[
        0,
        1,
        2,
        3,
        50256,
    ],  # it's crucial to have 50256 in the disabled_ids, otherwise we will not be able to distinguish the pad token id being part of merge or being just a padding token for merge
)

seq_len = 10_000
for tensor in tqdm(tokens.view(-1, seq_len)):
    ts_list = tensor.tolist()
    encoded_ts, _, _ = compressor.encode(ts_list)
    decoded_ts, _ = compressor.decode(encoded_ts)

    assert len(decoded_ts) == len(
        ts_list
    ), f"decoded length mismatch: {len(decoded_ts)} != {len(ts_list)}"
    assert decoded_ts == ts_list, "decoded content mismatch"
