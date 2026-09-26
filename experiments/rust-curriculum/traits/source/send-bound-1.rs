fn transferable<T: Send>(_: T) {}
fn main() { transferable(std::sync::Arc::new(4)); }
