fn main(){let x=std::rc::Rc::new(3u8);let h=std::thread::spawn(move||*x);let _=h.join();}
