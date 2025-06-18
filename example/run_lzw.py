from zip2zip_compression import LZWCompressor


compressor = LZWCompressor(
    initial_vocab_size=27,
    max_codebook_size=100,
    max_subtokens=5,
    pad_token_id=0,
    disabled_ids=[26],
)

text = "abcde" * 3

mapping = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
}

ids = [mapping[c] for c in text]

compressed_ids, attention_mask, codebook = compressor.encode(ids)


print(f"input ids: {ids}")
print(f"compressed_ids: {compressed_ids}")
print(f"attention_mask: {attention_mask}")
print(f"codebook: {codebook.to_dict()}")


# decompress

decompressed_ids = compressor.decode(compressed_ids)

print(f"decompressed_ids: {decompressed_ids}")
