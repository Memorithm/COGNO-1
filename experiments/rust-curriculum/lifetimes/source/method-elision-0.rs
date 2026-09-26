struct Selector;
impl Selector { fn choose(&self, value: &str) -> &str { value } }
fn main() { let _ = Selector.choose("ore"); }
