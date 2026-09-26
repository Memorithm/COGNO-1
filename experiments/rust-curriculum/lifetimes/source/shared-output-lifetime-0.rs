fn choose(a: &str, b: &str) -> &str { if a.len() > b.len() { a } else { b } }
fn main() { let _ = choose("a", "bb"); }
