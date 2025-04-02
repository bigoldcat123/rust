use std::{
    thread::{self},
    time::Duration,
};

fn main() {
    // thread::
    trpl::run(async {
        let task1 = async {
            for i in 0..10 {
                let thread_id = thread::current().id();
                println!("{},-> {:?}", i, thread_id);
                trpl::sleep(Duration::from_secs(1)).await;
            }
        };
        let task2 = async {
            for i in 0..5 {
                let thread_id = thread::current().id();
                println!("<{}>,-> {:?}", i, thread_id);
                trpl::sleep(Duration::from_secs(3)).await;
            }
        };  
        trpl::join!( task2);
        
        println!("{:?}","oVer");
        // let e = s.await.unwrap();
    });
}

fn show_thred_id() {
    println!("{:?}", thread::current().id());
}
