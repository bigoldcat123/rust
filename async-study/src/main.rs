use std::{
    thread::{self},
    time::Duration,
};

fn main() {
    // thread::
    trpl::run(async {
        let task1 = async {
            let thread_id = thread::current().id();
            for _ in 0..10 {
                trpl::sleep(Duration::from_secs(2)).await;
                println!("{:#?} {thread_id:?}", "hello from ");
            }
        };
        let task2 = async {
            let thread_id = thread::current().id();
            for _ in 0..10 {
                trpl::sleep(Duration::from_secs(1)).await;
                println!("{:#?} {thread_id:?}", "hello from ");
            }
        };
        // let res = trpl::join(task1, task2).await;
        let _ = trpl::race(task1, task2).await;

        // println!("{:#?}",);
        show_thred_id();
    });
}

fn show_thred_id() {
    println!("{:?}", thread::current().id());
}
