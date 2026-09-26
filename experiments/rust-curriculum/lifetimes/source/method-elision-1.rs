struct Selector;
impl Selector { fn choose<'a>(&self, value: &'a str) -> &'a str { value } }
fn main() { let _ = Selector.choose("ore"); }
