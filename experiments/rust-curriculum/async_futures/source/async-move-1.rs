fn make()->impl std::future::Future<Output=usize>{let s=String::from("future");async move{s.len()}} fn main(){}
