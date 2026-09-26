trait Produce { type Output; fn produce() -> Self::Output; }
struct Maker;
impl Produce for Maker { fn produce() -> u8 { 1 } }
fn main() { let _ = Maker::produce(); }
