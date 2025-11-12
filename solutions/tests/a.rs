#![allow(arithmetic_overflow)]

use std::collections::HashMap;

use solutions::every_day::give_me_random_array;



#[test]
fn hello() {
    let a = give_me_random_array(40, 10000, 0);
    println!("{a:?}");
}

#[test]
fn a() {
    let mut map = HashMap::new();
    map.insert("k", 1);
    println!("{}",map["k"]);
}
