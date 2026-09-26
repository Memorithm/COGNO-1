fn shared<T:Sync>(_:T){}fn main(){shared(std::sync::Mutex::new(1u8));}
