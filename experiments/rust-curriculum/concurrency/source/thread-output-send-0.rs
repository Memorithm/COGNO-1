fn main(){let h=std::thread::spawn(||std::rc::Rc::new(1u8));let _=h.join();}
