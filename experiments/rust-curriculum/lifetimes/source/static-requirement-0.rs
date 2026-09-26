fn view(value: &[u8]) -> &'static [u8] { value }
fn main() { let value = vec![1, 2]; let _ = view(&value); }
