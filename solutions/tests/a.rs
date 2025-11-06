
#![allow(arithmetic_overflow)]

use solutions::graph::num_ways;

#[test]
fn hello() {
    let res = num_ways(5,vec![vec![0,4],vec![1,2]],5);
    println!("{}", res);
}
