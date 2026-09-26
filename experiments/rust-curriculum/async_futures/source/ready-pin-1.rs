use std::future::Future;fn poll(f:&mut std::future::Ready<u8>,cx:&mut std::task::Context<'_>){let _=std::pin::Pin::new(f).poll(cx);}fn main(){}
