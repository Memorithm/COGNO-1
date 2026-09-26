fn change(value: &mut Box<i32>) { **value = 4; }
fn main() { let mut value = Box::new(1); change(&mut value); }
