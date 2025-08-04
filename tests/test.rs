mod utils;

use rand::{
    distr::{Distribution, Uniform},
    rng,
};
use std::time::{Duration, Instant};
use utils::get_tokens;
use zip2zip_compression::{
    codec::{Codebook, CompressionState},
    compressor::LZWCompressor,
    config::{CompressionConfig, PaddingStrategy},
    manager::CodebookManager,
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

    for chunk in ids.chunks(1024).take(2000).map(|chunk| chunk.to_vec()) {
        let (_, compression_state) =
            lzw_compressor.encode(&chunk, 0, PaddingStrategy::DoNotPad, false, None);

        let mut state = CompressionState::new_from_compressor(&lzw_compressor);
        let (updates, _) = codebook_manager.update_codebooks_and_get_updates(
            vec![chunk],
            vec![&mut state],
            true,
        );

        let encode_codebook = compression_state
            .codebook
            .to_list(true)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        assert_eq!(updates[0].len(), encode_codebook.len());

        assert_eq!(updates[0], encode_codebook);
    }
}

#[test]
fn benchmark_lzw_compressor() {
    let ids = get_tokens();

    let lzw_compressor = LZWCompressor::new(50257, 1024, 4, 50256, None);

    let mut encode_times = Vec::with_capacity(2000);
    let mut decode_times = Vec::with_capacity(2000);

    for chunk in ids.chunks(1024).take(2000).map(|chunk| chunk.to_vec()) {
        let start = Instant::now();

        let (compressed_ids, _) =
            lzw_compressor.encode(&chunk, 0, PaddingStrategy::DoNotPad, false, None);
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

    let mut codebook_manager = CodebookManager::new(compressor_config.clone());

    let mut times = Vec::with_capacity(2000);

    for chunk in ids.chunks(1024).take(2000).map(|chunk| chunk.to_vec()) {
        let start = Instant::now();
        let mut state = CompressionState::new(compressor_config.clone());
        codebook_manager.update_codebooks_and_get_updates(
            vec![chunk],
            vec![&mut state],
            true,
        );
        let end = Instant::now();
        times.push(end - start);
    }

    let mean_time = times.iter().sum::<Duration>() / times.len() as u32;
    println!("mean time: {:?}", mean_time);
}

fn simulate_generation(ids: Vec<usize>, codebook: Codebook) -> Vec<usize> {
    let mut rng = rng();
    let distr: Uniform<usize> = Uniform::try_from(0..2).unwrap();

    let mut output = Vec::new();

    for id in ids {
        if distr.sample(&mut rng) == 0 && id >= 50257 {
            let subtokens = codebook.get_subtokens(id).unwrap();
            output.extend_from_slice(&subtokens);
        } else {
            output.push(id);
        }
    }

    output
}

#[test]
fn test_codebook_manager_during_generation() {
    let ids = get_tokens()
        .iter()
        .map(|id| if *id == 50256 { 1 } else { *id })
        .collect::<Vec<_>>();

    let lzw_compressor = LZWCompressor::new(50257, 1024, 4, 50256, None);

    let mut codebook_manager = CodebookManager::new(lzw_compressor.config.clone());

    let prompt_len = 512;
    let sequence_len = 2048;

    for chunk in ids
        .chunks(sequence_len)
        .take(2000)
        .map(|chunk| chunk.to_vec())
    {
        let (compressed_chunk, encode_state) =
            lzw_compressor.encode(&chunk, 0, PaddingStrategy::DoNotPad, false, None);
        let (dc_chunk, _) = lzw_compressor.decode(&compressed_chunk[..prompt_len].to_vec());
        let (compressed_prompt, _) =
            lzw_compressor.encode(&dc_chunk, 0, PaddingStrategy::DoNotPad, false, None);

        let mut state = CompressionState::new(lzw_compressor.config.clone());
        codebook_manager.update_codebooks_and_get_updates(
            vec![compressed_prompt],
            vec![&mut state],
            false,
        );

        for t in simulate_generation(
            compressed_chunk[prompt_len..].to_vec(),
            encode_state.codebook.clone(),
        ) {
            codebook_manager.update_codebooks_and_get_updates(
                vec![vec![t]],
                vec![&mut state],
                false,
            );
        }

        let encode_hashmap = encode_state.codebook.to_list(false);
        let manager_hashmap = state.codebook.to_list(false);

        assert_eq!(encode_hashmap.len(), manager_hashmap.len());

        for (e, m) in encode_hashmap.iter().zip(manager_hashmap.iter()) {
            assert_eq!(e, m);
        }
    }
}
