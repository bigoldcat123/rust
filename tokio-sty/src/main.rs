use std::io::Write;

use bytes::{Buf, BufMut, BytesMut};
use env_logger::Env;
use log::{debug, error, info};
use serde::{Deserialize, Serialize};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let mut buf = BytesMut::with_capacity(1024);
    buf.put_slice(&[1, 2, 3, 3]);
    println!("{:?}",buf);
    let e = buf.get_u16();
    println!("{:?}",buf);    
    let x = u16::from_ne_bytes([1,0]);
    println!("{:?}",x);
    Ok(())
}
#[derive(Serialize, Deserialize, Debug)]
struct A {
    name: String,
}
