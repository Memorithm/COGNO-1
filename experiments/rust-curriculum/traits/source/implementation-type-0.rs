trait Read { fn read(&self) -> u32; }
struct Sensor;
impl Read for Sensor { fn read(&self) -> usize { 4 } }
fn main() { let _ = Sensor.read(); }
