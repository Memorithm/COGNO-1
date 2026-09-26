fn inspect(value: &String) -> usize { value.len() }
fn main() { let value = String::from("ore"); let size = inspect(&value); let _ = (size, value); }
