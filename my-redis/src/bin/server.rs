use my_redis::server::FileCheckerServer;
use tokio::io;

#[tokio::main]
async fn main() -> io::Result<()> {
    let server = FileCheckerServer::new("localhost:8848").await?;
    server.run().await.expect("msg");
    Ok(())
}
