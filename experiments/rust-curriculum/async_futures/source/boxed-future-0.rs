async fn read(){let f:Box<dyn std::future::Future<Output=u8>>=Box::new(async{1});let _=f.await;}fn main(){}
