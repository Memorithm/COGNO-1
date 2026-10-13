// Shared finite input corpus, not a numerical reference implementation.
// Dyadic values keep these small cases exactly representable in f32.
pub const LENGTHS: [usize; 6] = [1, 2, 7, 8, 17, 33];
pub const SEEDS: usize = 16;

pub fn fixture(len: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let left = (0..len)
        .map(|i| ((i * 7 + seed * 3) % 31) as f32 * 0.125 - 1.875)
        .collect();
    let right = (0..len)
        .map(|i| ((i * 11 + seed * 5) % 29) as f32 * 0.25 - 3.5)
        .collect();
    (left, right)
}

pub fn emit(op: &str, seed: usize, len: usize, values: &[f32]) {
    print!("{op}:{seed}:{len}");
    for value in values {
        assert!(value.is_finite(), "qualification output must be finite");
        print!(":{:08x}", value.to_bits());
    }
    println!();
}
