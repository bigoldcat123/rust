#![allow(arithmetic_overflow)]

use std::collections::HashMap;

use solutions::{eratosthenes, every_day::give_me_random_array};
#[test]
fn asda() {

    //1. x / 2 + x/2 + 1
    //2. x - 2 or x -1
    //3. x - 1 or x
    let (is,n) = eratosthenes(1000);
    for i in 0..100 {
        if is[i | (i + 1)] {
            println!("{} or {} = {}",i, i + 1, i | (i + 1));
        }
    }
}
macro_rules! 干 {
    (让 $a:ident 等于 $e:expr) => {
        let $a = $e;
    };
}
#[test]
fn feature() {
    let a = [1,1,1,1,1,3,3,3];
    let a = a.binary_search(&2);
    println!("{:?}",a);


}

#[test]
fn hello() {
    干!{让 你 等于 "一个神奇的数字"};
    println!("{} {} ",你,i32::MAX);


    let mut a = vec![];
    for _ in 0..10 {
        a.push(give_me_random_array(2, 1000, -1000));
    }
    println!("{:?}",a);
}

#[test]
fn a() {
    let mut map = HashMap::new();
    map.insert("k", 1);
    println!("{}",map["k"]);
}
