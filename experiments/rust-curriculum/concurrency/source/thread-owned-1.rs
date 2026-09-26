fn main(){let text=String::from("thread");let h=std::thread::spawn(move||text.len());let _=h.join();}
