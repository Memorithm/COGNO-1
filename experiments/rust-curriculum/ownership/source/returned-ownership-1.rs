fn duplicate(value: String) -> (String, String) { (value.clone(), value) }
fn main() { let _ = duplicate("ore".into()); }
