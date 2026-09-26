#[derive(Clone, Copy)]
struct Code(String);
fn main() { let value = Code("ore".into()); let _ = (value, value); }
