fn main(){let(tx,_rx)=std::sync::mpsc::channel::<u8>();let _=tx.send("two");}
