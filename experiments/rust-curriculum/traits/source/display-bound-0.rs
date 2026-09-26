fn show<T: std::fmt::Debug>(value: T) -> String { format!("{}", value) }
fn main() { let _ = show(3); }
