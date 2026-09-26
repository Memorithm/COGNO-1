struct Cell { left: i32, right: i32 }
fn main() { let mut c = Cell { left: 1, right: 2 }; let a = &mut c.left; let b = &mut c.right; *a += *b; }
