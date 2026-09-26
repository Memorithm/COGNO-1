fn transfer<T:Send>(_:T){}fn main(){transfer(std::sync::Arc::new(std::sync::Mutex::new(1u8)));}
