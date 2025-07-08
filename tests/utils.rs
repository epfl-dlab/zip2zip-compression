use hf_hub::api::sync::Api;
use std::fs::File;
use std::io::{BufReader, Read};

pub fn get_tokens() -> Vec<usize> {
    let api = Api::new().unwrap();

    let repo = api.dataset("kjj0/fineweb10B-gpt2".to_string());
    let filename = repo.get("fineweb_train_000001.bin").unwrap();

    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);

    let mut header_buffer = vec![0u8; 256 * 4];
    reader.read_exact(&mut header_buffer).unwrap();
    let header: Vec<i32> = header_buffer
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut ids_buffer = Vec::new();
    reader.read_to_end(&mut ids_buffer).unwrap();

    let ids: Vec<usize> = ids_buffer
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as usize)
        .collect();

    assert!(
        header[0] == 20240520,
        "magic number mismatch in the data .bin file"
    );
    assert!(header[1] == 1, "unsupported version");
    
    let num_tokens = header[2] as usize;
    assert!(num_tokens == ids.len(), "number of tokens mismatch");

    ids
}
