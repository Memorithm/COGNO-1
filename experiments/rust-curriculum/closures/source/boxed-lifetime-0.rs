fn make()->Box<dyn Fn()->usize>{let s=String::from("boxed");Box::new(||s.len())} fn main(){}
