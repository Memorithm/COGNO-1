fn length<T: AsRef<str> + ?Sized>(value: &T) -> usize { value.as_ref().len() }
fn main() { let _ = length("ore"); }
