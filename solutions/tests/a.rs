#![allow(arithmetic_overflow)]

use std::collections::HashMap;

use solutions::every_day::give_me_random_array;
#[test]
fn asda() {
    fn a(l:&[i32],s:&mut Vec<i32>,len:usize,start:usize) {
        if s.len() == len {
            println!("{:?}",s);
            return;
        }
        for i in start..l.len() {
            s.push(l[i]);
            a(l,s,len,i+1);
            s.pop();
        }
    }
    a(&[1,2,3,4,5],&mut vec![],2,0);
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
