mod utils;

use tqdm::Iter;
use utils::get_tokens;
use std::collections::{HashMap, HashSet};
use zip2zip_compression::{Codebook, CompressionConfig, LZWCompressor, PaddingStrategy};

fn get_alphabet_compression_config() -> CompressionConfig {
    let mut disabled_ids: HashSet<usize> = HashSet::new();
    disabled_ids.insert(26); // 'z'
    disabled_ids.insert(0); // '\0' padding token

    // 26 letters + 1 for the pad token
    CompressionConfig::new(27, 100, 5, 0, disabled_ids)
}

fn get_base_letter_to_id_map() -> HashMap<char, usize> {
    let mut base_letter_to_id_map = HashMap::new();
    for (i, letter) in "abcdefghijklmnopqrstuvwxyz".chars().enumerate() {
        base_letter_to_id_map.insert(letter, i + 1);
    }
    // add \0 for the pad token at position 0
    base_letter_to_id_map.insert('\0', 0);
    base_letter_to_id_map
}

#[test]
fn test_alphabet_codebook_config() {
    let alphabet_codebook_config = get_alphabet_compression_config();

    assert_eq!(alphabet_codebook_config.initial_vocab_size, 27);
    assert_eq!(alphabet_codebook_config.max_codebook_size, 100);
    assert_eq!(alphabet_codebook_config.max_subtokens, 5);
    assert_eq!(alphabet_codebook_config.pad_token_id, 0);
}

#[test]
fn test_alphabet_codebook() {
    let alphabet_config = get_alphabet_compression_config();

    let mut alphabet_codebook = Codebook::new(&alphabet_config);

    //insert a few entries
    alphabet_codebook.insert(vec![1, 2], 0);
    alphabet_codebook.insert(vec![2, 3], 1);
    alphabet_codebook.insert(vec![3, 4, 5], 2);

    //get a hyper_id
    let hyper_id = alphabet_codebook.get(&vec![1, 2]).unwrap();
    assert_eq!(*hyper_id, 27);

    //get a hyper_id
    let hyper_id = alphabet_codebook.get(&vec![3, 4, 5]).unwrap();
    assert_eq!(*hyper_id, 29);

    //check contains key
    assert!(alphabet_codebook.contains_key(&vec![1, 2]));

    //get the base ids
    let base_ids = alphabet_codebook.get_subtokens(27).unwrap();
    assert_eq!(base_ids, vec![1, 2]);

    //get the base ids
    let base_ids = alphabet_codebook.get_subtokens(29).unwrap();
    assert_eq!(base_ids, vec![3, 4, 5]);
}

#[test]
fn test_lzw_compressor() {
    let alphabet_config = get_alphabet_compression_config();

    let lzw_compressor = LZWCompressor {
        config: alphabet_config,
    };

    // encode a simple sentence
    let ids = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5];
    let (compressed_ids, state) =
        lzw_compressor.encode(&ids, 0, PaddingStrategy::DoNotPad, false, None);
    assert_eq!(compressed_ids, vec![1, 2, 3, 4, 5, 27, 29, 31, 28, 30]);
    assert!(state.codebook.inner.len() < ids.len() - 1);

    let alphabet_config = get_alphabet_compression_config();

    let lzw_compressor2 = LZWCompressor::new(
        alphabet_config.initial_vocab_size,
        alphabet_config.max_codebook_size,
        alphabet_config.max_subtokens,
        alphabet_config.pad_token_id,
        Some(vec![26]),
    );

    //  now decode the compressed ids
    let (decoded_ids, _) = lzw_compressor2.decode(&compressed_ids);
    assert_eq!(decoded_ids, ids);
}

#[test]
fn test_encode_decode() {
    let ids = get_tokens();

    let lzw_compressor = LZWCompressor::new(
        50257,
        1024,
        4,
        50256,
        None,
    );

    for chunk_size in [4096, 2048, 1024, 512, 256, 128, 64, 32] {
        for chunk in ids.chunks(chunk_size).take(2000).map(|chunk| chunk.to_vec()).tqdm() {
            let (compressed_ids, _) = lzw_compressor.encode(&chunk, 0, PaddingStrategy::DoNotPad, false, None);
            let (decoded_ids, _) = lzw_compressor.decode(&compressed_ids);
            assert_eq!(decoded_ids, chunk);
        }
    }
}
