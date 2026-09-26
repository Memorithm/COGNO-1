fn main(){let h=std::thread::spawn(||std::sync::Arc::new(1u8));let _=h.join();}
