trait Read { fn read(&self) -> u32; }
struct Sensor;
impl Read for Sensor { fn read(&self) -> u32 { 4 } }
fn main() { let _ = Sensor.read(); }
