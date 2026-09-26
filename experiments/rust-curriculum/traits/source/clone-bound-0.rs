fn copy<T>(value: &T) -> T { value.clone() }
fn main() { let _ = copy(&String::from("ore")); }
