use std::{cmp::Ordering, collections::HashMap, f32::consts::PI, future, thread, time::Duration};

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
    fn get_dicimal(mut left: i32, right: i32) {
        let mut map = HashMap::new();
        loop {
            left = left % right;
            if left == 0 {
                break;
            }
            if map.contains_key(&left) {
                break;
            }
            map.insert(left, 0);
            left *= 10;
            let x = (left / right) as i64;
            println!("x{:?}", x);
        }
    }
    get_dicimal(1, 1000);
}

#[test]
fn convert_to_title() {
    A::largest_number(vec![432, 43243]);

    let mut e = vec![34323,3432];
    e.sort_by(|a, b| {
        let mut e = String::new();
        e.push_str(&a.to_string());
        e.push_str(&b.to_string());
        let mut x = String::new();
        x.push_str(&b.to_string());
        x.push_str(&a.to_string());
        let left = e.parse::<i32>().unwrap();
        let right = x.parse::<i32>().unwrap();
        if left > right {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });
    println!("e{:?}",e);
}


#[test]
fn sort() {
    for i in -10..0 {
        println!("{:?}",i);
    }
}