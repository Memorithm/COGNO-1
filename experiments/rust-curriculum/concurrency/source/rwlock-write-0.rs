fn main(){let lock=std::sync::RwLock::new(3u8);*lock.read().unwrap()=4;}
