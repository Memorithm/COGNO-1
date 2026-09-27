//! Explicitly budgeted, hash-bound research training; no model promotion.
#![forbid(unsafe_code)]
mod train_v2_support;
use train_v2_support::*;
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("plan") {
        return Err("usage: rust_train_v2 plan PROTOCOL PROTOCOL_SHA CORPUS PROVENANCE".into());
    }
    let admission = admit(&args[1..])?;
    print!("{}", plan(&admission));
    Ok(())
}
