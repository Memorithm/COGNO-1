fn send<T:Send>(_:T){} fn main(){let x=std::sync::Arc::new(1);send(move||drop(x));}
