fn values() -> impl Iterator<Item = &'static i32> { let data = vec![1, 2]; data.iter() }
fn main() { let _ = values(); }
