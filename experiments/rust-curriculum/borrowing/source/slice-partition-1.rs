fn main() { let mut data = [1, 2]; let (left, right) = data.split_at_mut(1); left[0] += right[0]; }
