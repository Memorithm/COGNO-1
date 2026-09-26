fn main(){let x=std::sync::Mutex::new(6u8);let _:u8=*x.lock().unwrap();}
