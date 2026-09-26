fn make() -> impl Fn() -> usize { let value = String::from("ore"); move || value.len() }
fn main() { let _ = make(); }
