#[derive(Clone, Copy)]
struct Code(u32);
fn main() { let value = Code(2); let _ = (value, value); }
