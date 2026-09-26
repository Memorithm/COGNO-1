fn make() -> &'static str { let value = String::from("ore"); &value }
fn main() { let _ = make(); }
