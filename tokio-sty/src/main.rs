use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let res = reqwest::get("http://127.0.0.1:3000").await?;
    let mut e = res.bytes_stream();
    while let Some(chunk) = e.next().await {
        match chunk {
            Ok(bytes) => {
               println!("{:?}",bytes.len());
            }
            Err(_) => {}
        }
    }
    Ok(())
}
#[derive(Serialize, Deserialize, Debug)]
struct A {
    name: String,
}
