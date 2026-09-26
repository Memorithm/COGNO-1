fn main() { let view; { let value = String::from("ore"); view = &value; } let _ = view.len(); }
