fn main() { let value = Some(String::from("ore")); let part = value.as_ref().unwrap(); let _ = (part.len(), value.is_some()); }
