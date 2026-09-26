fn main() { let mut value = 3; let read = &value; let write = &mut value; *write += 1; let _ = *read; }
