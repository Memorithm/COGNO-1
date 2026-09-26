fn create<T: Default>() -> T { T::default() }
fn main() { let _: i32 = create(); }
