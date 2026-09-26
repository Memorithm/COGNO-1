fn main() { let value = String::from("ore"); let view; { view = &value; } let _ = view.len(); }
