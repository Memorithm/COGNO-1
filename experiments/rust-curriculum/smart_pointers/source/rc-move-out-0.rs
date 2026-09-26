fn main(){let x=std::rc::Rc::new(String::from("shared"));let value:String=*x;drop(value);}
