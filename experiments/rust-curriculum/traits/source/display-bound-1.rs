fn show<T: std::fmt::Display>(value: T) -> String { format!("{}", value) }
fn main() { let _ = show(3); }
