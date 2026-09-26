fn main(){let owner=std::rc::Rc::new(String::from("weak"));let weak=std::rc::Rc::downgrade(&owner);let _=weak.len();}
