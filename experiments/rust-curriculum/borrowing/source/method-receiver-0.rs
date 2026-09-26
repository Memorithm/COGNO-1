struct Counter(i32);
impl Counter { fn bump(&mut self) { self.0 += 1; } }
fn main() { let counter = Counter(0); counter.bump(); }
