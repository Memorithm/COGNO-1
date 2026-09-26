fn smaller<T>(a: T, b: T) -> bool where T: PartialOrd { a < b }
fn main() { let _ = smaller(1, 2); }
