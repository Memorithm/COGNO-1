fn shared<T:Sync>(_:T){}fn main(){shared(std::cell::Cell::new(1u8));}
