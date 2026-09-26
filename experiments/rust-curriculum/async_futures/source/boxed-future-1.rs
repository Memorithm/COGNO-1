async fn read(){let f:std::pin::Pin<Box<dyn std::future::Future<Output=u8>>>=Box::pin(async{1});let _=f.await;}fn main(){}
