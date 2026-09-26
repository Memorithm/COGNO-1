fn main(){let x=std::sync::Arc::new(3u8);let h=std::thread::spawn(move||*x);let _=h.join();}
