fn transfer<T:Send>(_:T){}fn main(){transfer(std::sync::Arc::new(std::cell::RefCell::new(1u8)));}
