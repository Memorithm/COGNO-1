fn main(){let n=std::sync::atomic::AtomicUsize::new(0);n.store(true,std::sync::atomic::Ordering::Relaxed);}
