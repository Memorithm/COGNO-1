fn length<T: AsRef<str>>(value: &T) -> usize { value.as_ref().len() }
fn main() { let _ = length("ore"); }
