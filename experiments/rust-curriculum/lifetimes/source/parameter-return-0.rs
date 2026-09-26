fn view(value: &str) -> &str { let copy = value.to_owned(); &copy }
fn main() { let value = String::from("ore"); let _ = view(&value); }
