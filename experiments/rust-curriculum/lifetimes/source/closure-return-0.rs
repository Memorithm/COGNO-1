fn make() -> impl Fn() -> usize { let value = String::from("ore"); || value.len() }
fn main() { let _ = make(); }
