fn main() { let value = Some(String::from("ore")); let part = value.unwrap(); let _ = (part.len(), value.is_some()); }
