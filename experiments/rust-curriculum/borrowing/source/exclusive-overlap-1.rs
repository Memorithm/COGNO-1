fn main() { let mut value = 3; let first = &mut value; *first += 1; let second = &mut value; *second += 1; }
