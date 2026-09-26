fn main(){let(tx,_rx)=std::sync::mpsc::channel();let s=String::from("message");let _=tx.send(s);drop(s);}
