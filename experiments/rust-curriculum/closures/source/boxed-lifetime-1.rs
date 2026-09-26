fn make()->Box<dyn Fn()->usize>{let s=String::from("boxed");Box::new(move||s.len())} fn main(){}
