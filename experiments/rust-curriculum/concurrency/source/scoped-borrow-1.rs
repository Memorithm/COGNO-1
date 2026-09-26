fn main(){let data=vec![1u8,2];std::thread::scope(|scope|{scope.spawn(||data.len());});}
