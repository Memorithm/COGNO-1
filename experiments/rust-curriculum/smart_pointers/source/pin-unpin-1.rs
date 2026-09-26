fn main(){let mut x=Box::pin(7u8);let _: &mut u8=x.as_mut().get_mut();}
