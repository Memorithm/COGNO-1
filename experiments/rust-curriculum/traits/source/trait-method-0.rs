trait Measure { fn size(&self) -> usize; }
struct Sample;
impl Measure for Sample {}
fn main() { let _ = Sample.size(); }
