fn values() -> impl Iterator<Item = i32> { let data = vec![1, 2]; data.into_iter() }
fn main() { let _ = values(); }
