fn consume(_: String) {}
fn main() { let value = String::from("ore"); for _ in 0..2 { consume(value.clone()); } }
