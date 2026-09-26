#![allow(dead_code)]
static mut Y: u32 = 0;
unsafe fn should_ok() {
    Y = 1;
}
fn main() {}
