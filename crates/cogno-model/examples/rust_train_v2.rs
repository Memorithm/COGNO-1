//! Explicitly budgeted, hash-bound research training; no model promotion.
#![forbid(unsafe_code)]
mod train_v2_support;
use train_v2_support::*;
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("plan") => { let a = admit(&args[1..])?; print!("{}", plan(&a)); }
        Some("train") if args.len() == 6 => { let a = admit(&args[1..5])?; train(&a, std::path::Path::new(&args[5]))?; }
        Some("verify") if args.len() == 7 => { let a = admit(&args[1..5])?; verify(&a, std::path::Path::new(&args[5]), &args[6])?; println!("bundle verified"); }
        Some("select") if args.len() == 8 => { let a = admit(&args[1..5])?; select(&a, std::path::Path::new(&args[5]), &args[6], std::path::Path::new(&args[7]))?; }
        Some("test") if args.len() == 9 => { let a = admit(&args[1..5])?; test_selected(&a, std::path::Path::new(&args[5]), &args[6], std::path::Path::new(&args[7]), std::path::Path::new(&args[8]))?; }
        _ => return Err("usage: rust_train_v2 plan PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE | train PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE NEW_OUTPUT_DIR".into()),
    }
    Ok(())
}
