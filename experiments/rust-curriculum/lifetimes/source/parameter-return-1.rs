fn view(value: &str) -> &str { value }
fn main() { let value = String::from("ore"); let _ = view(&value); }
