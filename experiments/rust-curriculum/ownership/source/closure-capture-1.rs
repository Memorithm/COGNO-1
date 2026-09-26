fn main() { let name = String::from("ore"); let read = || name.len(); let _ = (name.len(), read()); }
