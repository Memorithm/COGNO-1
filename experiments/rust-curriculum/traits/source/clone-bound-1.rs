fn copy<T: Clone>(value: &T) -> T { value.clone() }
fn main() { let _ = copy(&String::from("ore")); }
