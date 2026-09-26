fn main(){let n=std::sync::atomic::AtomicUsize::new(0);n.store(1usize,std::sync::atomic::Ordering::Relaxed);}
