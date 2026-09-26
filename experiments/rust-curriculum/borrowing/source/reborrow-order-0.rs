fn main() { let mut value = 1; let parent = &mut value; let child = &mut *parent; *parent += 1; *child += 1; }
