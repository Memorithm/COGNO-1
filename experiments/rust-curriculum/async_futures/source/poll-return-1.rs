struct Ready;impl std::future::Future for Ready{type Output=u8;fn poll(self:std::pin::Pin<&mut Self>,_:&mut std::task::Context<'_>)->std::task::Poll<u8>{std::task::Poll::Ready(1)}}fn main(){}
