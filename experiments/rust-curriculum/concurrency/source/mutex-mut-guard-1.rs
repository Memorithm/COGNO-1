fn main(){let lock=std::sync::Mutex::new(3u8);let mut guard=lock.lock().unwrap();*guard+=1;}
