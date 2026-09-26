struct View<'a> { value: &'a str }
fn main() { let value = View { value: "ore" }; let _ = value.value; }
