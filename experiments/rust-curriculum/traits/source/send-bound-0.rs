fn transferable<T: Send>(_: T) {}
fn main() { transferable(std::rc::Rc::new(4)); }
