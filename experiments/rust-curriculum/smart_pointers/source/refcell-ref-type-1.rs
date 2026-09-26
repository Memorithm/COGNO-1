fn main(){let x=std::cell::RefCell::new(5u8);let guard=x.borrow();let _: &u8=&guard;}
