fn main() { let mut value = 3; let first = &mut value; let second = &mut value; *first += 1; *second += 1; }
