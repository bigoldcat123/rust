#![allow(arithmetic_overflow)]

use std::collections::HashMap;

use solutions::every_day::give_me_random_array;

macro_rules! 干 {
    (让 $a:ident 等于 $e:expr) => {
        let $a = $e;
    };
}
#[test]
fn feature() {
    let a = [1,2,3];
    let b = [2,-1,-100];
    println!("{}",a > b);

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
