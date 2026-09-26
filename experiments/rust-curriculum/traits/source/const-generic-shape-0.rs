fn length<const N: usize>(_: [u8; N]) -> usize { N }
fn main() { let _ = length::<4>([0; 3]); }
