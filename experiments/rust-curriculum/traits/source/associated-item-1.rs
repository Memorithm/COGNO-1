fn sum<I: Iterator<Item = i32>>(iter: I) -> i32 { iter.fold(0, |a, b| a + b) }
fn main() { let _ = sum([1, 2].into_iter()); }
