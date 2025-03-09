use std::{io, process};

use my_redis::{FileChecker, parse_args};
fn main() -> io::Result<()> {
    let addr = parse_args();
    let checker = FileChecker::new(addr)?;
    if let Err(err) = checker.run() {
        eprintln!("{:?}", err);
        process::exit(0);
    }
    Ok(())
}
