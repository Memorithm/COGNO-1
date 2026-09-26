fn choose<'a>(a: &'a str, b: &'a str) -> &'a str { if a.len() > b.len() { a } else { b } }
fn main() { let _ = choose("a", "bb"); }
