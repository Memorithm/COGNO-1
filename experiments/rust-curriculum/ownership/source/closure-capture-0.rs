fn main() { let name = String::from("ore"); let read = move || name.len(); let _ = (name.len(), read()); }
