use std::{collections::HashMap, f32::consts::PI, future, thread, time::Duration};

use solutions::A;

#[test]
fn test_function() {
    let e = A::generate(5);
    println!("{:#?}", e);
}

#[test]
fn ele() {
    let a = ["aa", "b"];
    let s = "a".to_string();
    println!("{:?}", a.contains(&&s[..]));
}

#[test]
fn elee() {
    fn get_dicimal(mut left:i32,right:i32) {
        let mut map = HashMap::new();
        loop {
             left = left % right ;
             if left == 0{
                break;
             }
             if map.contains_key(&left) {
                break;
             }
             map.insert(left, 0);
             left *= 10;
             let x = (left / right) as i64;
             println!("x{:?}",x);
        }
    }
    get_dicimal(1, 1000);
}
