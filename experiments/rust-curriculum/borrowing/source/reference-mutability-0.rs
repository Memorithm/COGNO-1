fn update(value: &i32) { *value += 1; }
fn main() { let mut value = 2; update(&mut value); }
