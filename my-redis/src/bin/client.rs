
use std::{io, process};

use my_redis::{parse_args, FileChecker};
fn main() -> io::Result<()>{
    let addr = parse_args();
    let checker = FileChecker::new(addr)?;
    if let Err(err) = checker.run() {
        eprintln!("{:?}",err);
        process::exit(0);
    }
    Ok(())
}