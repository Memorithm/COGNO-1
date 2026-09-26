fn main() { let mut value = 3; let read = &value; let _ = *read; let write = &mut value; *write += 1; }
