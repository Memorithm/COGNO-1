fn send<T:Send>(_:T){} fn main(){let x=std::rc::Rc::new(1);send(move||drop(x));}
