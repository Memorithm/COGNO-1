fn main() { let mut data = vec![1, 2]; for value in &data { let _ = (*value, data.len()); } data.push(3); }
