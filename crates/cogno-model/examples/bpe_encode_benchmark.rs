//! Synthetic CPU comparison against the previous sequential merge implementation.
//! Run with cargo run --release -p cogno-model --example bpe_encode_benchmark.
use cogno_model::bpe_tokenizer::BpeTokenizer;
use std::{hint::black_box, time::Instant};

fn reference(input: &[u8], rules: &[(u16, u16)]) -> Vec<u16> {
    let mut ids: Vec<_> = input.iter().map(|&b| u16::from(b)).collect();
    for (rank, &pair) in rules.iter().enumerate() {
        let mut read = 0;
        let mut write = 0;
        while read < ids.len() {
            if read + 1 < ids.len() && (ids[read], ids[read + 1]) == pair {
                ids[write] = (259 + rank) as u16;
                read += 2;
            } else {
                ids[write] = ids[read];
                read += 1;
            }
            write += 1;
        }
        ids.truncate(write);
    }
    let mut out = vec![256];
    out.extend(ids);
    out.push(257);
    out
}
fn main() {
    let train = b"fn main() { let value = 42; assert_eq!(value, 42); }\n".repeat(4);
    let tokenizer = BpeTokenizer::train(&[&train], 384, 512).unwrap();
    let artifact = tokenizer.to_bytes();
    let rules: Vec<_> = artifact[12..]
        .chunks_exact(4)
        .map(|p| {
            (
                u16::from_le_bytes([p[0], p[1]]),
                u16::from_le_bytes([p[2], p[3]]),
            )
        })
        .collect();
    println!("case,input_bytes,iterations,reference_ns,optimized_ns");
    for (name, input) in [
        ("repeated_rust", train),
        ("unseen_binary", (0u8..=255).collect()),
        ("overlap", vec![b'a'; 256]),
    ] {
        assert_eq!(tokenizer.encode(&input).unwrap(), reference(&input, &rules));
        let iterations = 20_000;
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(reference(black_box(&input), black_box(&rules)));
        }
        let old = start.elapsed().as_nanos() / iterations;
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(tokenizer.encode(black_box(&input)).unwrap());
        }
        let new = start.elapsed().as_nanos() / iterations;
        println!("{name},{},{iterations},{old},{new}", input.len());
    }
}
