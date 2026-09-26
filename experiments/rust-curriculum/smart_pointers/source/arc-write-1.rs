fn main(){let x=std::sync::Arc::new(std::sync::Mutex::new(1));*x.lock().unwrap()=2;}
