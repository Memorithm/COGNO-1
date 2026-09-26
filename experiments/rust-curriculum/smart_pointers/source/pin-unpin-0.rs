fn main(){let mut x=Box::pin(std::marker::PhantomPinned);let _=x.as_mut().get_mut();}
