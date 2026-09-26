struct View<'a>(&'a str);
fn main() { let text = String::from("ore"); let view; { view = View(&text); } let _ = view.0.len(); }
