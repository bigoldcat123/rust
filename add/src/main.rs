use std::{env, process};

use log::info;

fn main() {
    env_logger::init();
    let args = env::args().collect::<Vec<String>>();
    info!("args:{:?}", args);
    if args.len() != 3 {
        log::error!("arg must be 2 !");
        return;
    }
    let left = match args[1].parse::<i32>() {
        Ok(x) => x,
        Err(_) => {
            log::error!("must be a number! {} is not a number !", args[1]);
            process::exit(1);
        }
    };
    let right = match args[2].parse::<i32>() {
        Ok(x) => x,
        Err(_) => {
            log::error!("must be a number! {} is not a number !", args[2]);
            process::exit(1);
        }
    };

    println!("{} + {} = {}", left, right, left + right);
}
