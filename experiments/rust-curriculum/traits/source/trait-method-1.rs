trait Measure { fn size(&self) -> usize; }
struct Sample;
impl Measure for Sample { fn size(&self) -> usize { 1 } }
fn main() { let _ = Sample.size(); }
