struct View<'a>(&'a str);
fn main() { let view; { let text = String::from("ore"); view = View(&text); } let _ = view.0.len(); }
