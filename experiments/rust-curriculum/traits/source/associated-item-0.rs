fn sum<I: Iterator>(iter: I) -> i32 { iter.fold(0, |a, b| a + b) }
fn main() { let _ = sum([1, 2].into_iter()); }
