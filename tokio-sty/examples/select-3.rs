use std::io::{self, Error};

use tokio::{
    main,
    net::TcpListener,
    select,
    sync::oneshot,
    time::{Instant},
};

#[main]
async fn main() -> Result<(), Error> {
    let now = Instant::now();
    let (tx, rx) = oneshot::channel();

    tokio::spawn(async {
        // time::sleep(Duration::from_secs(1)).await;
        tx.send(()).unwrap();
    });

    let server = TcpListener::bind("localhost:8848").await?;

    let res = select! {
        res = async move {
            loop {
                let socket = server.accept().await?;
                println!("{:?}",socket);
            }
            //ignore
            Ok::<_, io::Error>("end")
        } => {res},
        _ = rx => {
            println!("G");
            Ok::<_, io::Error>("end2")
        }
    };
    println!("res => {res:?}",);
    // println!("{:?}",server);
    let elapsed = now.elapsed();
    println!("elapsed: {:?}", elapsed);
    Ok(())
}
