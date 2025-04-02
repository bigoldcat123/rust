use std::{
    fmt::format,
    pin::{Pin, pin},
    process::Output,
    thread,
    time::Duration,
};

use futures::{
    SinkExt, Stream, StreamExt, TryFutureExt,
    channel::oneshot::channel,
    future::{err, join_all, ok},
    stream,
};

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let task1 = async {
        println!("{:?}", "task1 start");
        for i in 0..3 {
            let current_id = thread::current().id();
            tokio::time::sleep(Duration::from_secs(1)).await;
            println!("{:?},{:?}", i, current_id);
        }
        Peope {}
    };

    let task2 = pin!(async {
        for i in 0..6 {
            let current_id = thread::current().id();
            tokio::time::sleep(Duration::from_millis(100)).await;
            println!("{:?}---{:?}", i, current_id);
        }
        Peope {}
    });
    let task1 = pin!(task1);
    let futures: Vec<Pin<&mut dyn Future<Output = Peope>>> = vec![task1, task2];
    // join_all(futures).await;

    let mut res = stream::iter(futures);
    while let Some(x) = res.next().await {
        // println!("{:?}",x.await);
        let prople = x.await;
        prople.do_some();
    }
    let a = 100;
    let res = set_timeout(
        async move {
            println!("{:?}", a);
            tokio::time::sleep(Duration::from_millis(100)).await;
            2
        },
        Duration::from_millis(300),
    )
    .await
    .unwrap_or(1);
    println!("res = > {:?}", res);

    let mut messages = get_messages();
    while let Ok(x) = set_timeout(messages.next(), Duration::from_millis(100)).await {
        println!("x-> {:?}", x);
    }

    // tokio::join!(futures);
    //
    // tokio::time::sleep(Duration::from_secs(100)).await;
}

async fn set_timeout<T>(work: impl Future<Output = T>, duration: Duration) -> Result<T, String> {
    tokio::select! {
        r  = work => {
            ok(r)
        }
        _ = async {
            tokio::time::sleep(duration).await
        } => {
            err("nonon".to_string())
        }
    }
    .await
}

fn get_messages() -> impl Stream<Item = String> {
    // tokio::stream

    let (mut tx, rx) = futures::channel::mpsc::channel(200);
    tokio::spawn(async move {
        for i in 0..10 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            tx.send(format!("hello the idx is {}", i)).await.unwrap();
        }
    });
    rx
}

trait DoSomething {
    fn do_some(self);
}
struct Peope {}
impl DoSomething for Peope {
    fn do_some(self) {
        println!("{:?}", "i am a people i am doing something!");
    }
}
