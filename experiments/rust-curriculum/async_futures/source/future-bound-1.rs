fn accept(_:impl std::future::Future<Output=u8>){} fn main(){accept(async{8u8});}
