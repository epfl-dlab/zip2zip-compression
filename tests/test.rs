mod utils;

use std::time::{Duration, Instant};
use tqdm::Iter;
use utils::get_tokens;
use zip2zip_compression::{
    Codebook, CodebookManager, CompressionConfig, LZWCompressor, PaddingStrategy,
};

fn get_alphabet_compression_config() -> CompressionConfig {
    // 26 letters + 1 for the pad token
    CompressionConfig::new(27, 100, 5, 0, Some(vec![26, 0]))
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

    let lzw_compressor = LZWCompressor::new(50257, 1024, 4, 50256, None);

    for chunk_size in [4096, 2048, 1024, 512, 256, 128, 64, 32] {
        for chunk in ids
            .chunks(chunk_size)
            .take(2000)
            .map(|chunk| chunk.to_vec())
            .tqdm()
        {
            let (compressed_ids, _) =
                lzw_compressor.encode(&chunk, 0, PaddingStrategy::DoNotPad, false, None);
            let (decoded_ids, _) = lzw_compressor.decode(&compressed_ids);
            assert_eq!(decoded_ids, chunk);
        }
    }
}

#[test]
fn test_codebook_manager() {
    let ids = get_tokens();

    let lzw_compressor = LZWCompressor::new(50257, 1024, 4, 50256, None);

    let mut codebook_manager = CodebookManager::new(lzw_compressor.config.clone());

    for chunk in ids
        .chunks(1024)
        .take(2000)
        .map(|chunk| chunk.to_vec())
        .tqdm()
    {
        let (_, compression_state) =
            lzw_compressor.encode(&chunk, 0, PaddingStrategy::DoNotPad, false, None);

        let (updates, _) = codebook_manager.update_codebooks(vec![chunk]);

        let encode_codebook = compression_state.codebook.to_list(true).into_iter().flatten().collect::<Vec<_>>();

        assert_eq!(
            updates[0].len(),
            encode_codebook.len()
        );

        assert_eq!(updates[0], encode_codebook);

        codebook_manager.reset();
    }
}

#[test]
fn benchmark_lzw_compressor() {
    let ids = get_tokens();

    let lzw_compressor = LZWCompressor::new(50257, 1024, 4, 50256, None);

    let mut encode_times = Vec::with_capacity(2000);
    let mut decode_times = Vec::with_capacity(2000);

    for chunk in ids.chunks(1024).take(2000).map(|chunk| chunk.to_vec()).tqdm() {
        let start = Instant::now();

        let (compressed_ids, _) = lzw_compressor.encode(&chunk, 0, PaddingStrategy::DoNotPad, false, None);
        let end = Instant::now();
        encode_times.push(end - start);

        let start = Instant::now();
        let (_, _) = lzw_compressor.decode(&compressed_ids);
        let end = Instant::now();
        decode_times.push(end - start);
    }

    let mean_encode_time = encode_times.iter().sum::<Duration>() / encode_times.len() as u32;
    let mean_decode_time = decode_times.iter().sum::<Duration>() / decode_times.len() as u32;

    println!("mean encode time: {:?}", mean_encode_time);
    println!("mean decode time: {:?}", mean_decode_time);
}

#[test]
fn benchmark_codebook_manager() {
    let ids = get_tokens();

    let compressor_config = CompressionConfig::new(50257, 1024, 4, 50256, None);

    let mut codebook_manager = CodebookManager::new(compressor_config);

    let mut times = Vec::with_capacity(2000);

    for chunk in ids.chunks(1024).take(2000).map(|chunk| chunk.to_vec()).tqdm() {
        let start = Instant::now();
        let (_, _) = codebook_manager.update_codebooks(vec![chunk]);
        let end = Instant::now();
        times.push(end - start);

        codebook_manager.reset();
    }

    let mean_time = times.iter().sum::<Duration>() / times.len() as u32;
    println!("mean time: {:?}", mean_time);
}
