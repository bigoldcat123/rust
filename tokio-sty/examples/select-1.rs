use std::{thread, time::Duration};

use tokio::select;

#[tokio::main]
async fn main() {
    select! {
        var = task1() => {
            println!("task1 is done");
            println!("{:?}",var);
        }
        var = task2() => {
            println!("task2 is done");
            println!("{:?}",var);
        }
    }
}

async fn task1() -> i32 {
    let id = thread::current().id();
    println!("current thread: {:?}", id);
    tokio::time::sleep(Duration::from_secs(1)).await;
    return 1;
}

async fn task2() -> i32 {
    let id = thread::current().id();
    println!("current thread: {:?}", id);
    tokio::time::sleep(Duration::from_secs(1)).await;
    return 2;
}
