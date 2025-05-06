use std::{
    cmp::Ordering,
    collections::HashMap,
    f32::consts::PI,
    future,
    rc::Rc,
    thread,
    time::{Duration, Instant},
};

use solutions::{A, TreeNode};

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

    let mut e = vec![34323, 3432];
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
    println!("e{:?}", e);
}

#[test]
fn sort() {
    let val = vec![1, 2, 3, 4, 5, 5, 6, 1,2, 3, 4, 5, 5, 6, 1,2, 3, 4, 5, 5, 6, 1, 5, 6, 1,2, 3, 4];
    let val2 = val.clone();
    let wei = vec![2, 3, 4, 3, 4, 6, 6, 1,2, 3, 4,2, 3, 4, 5, 5, 6, 1, 5, 6, 1,2, 3, 4, 5, 5, 6, 1];
    let wei2 = wei.clone();
    let cap = wei.iter().sum::<i32>() / 2;

    let ist = std::time::Instant::now();
    let e = A::knapsack_(val, wei, cap);
    let d = ist.elapsed();
    println!("{:?}", d);
    let ist = Instant::now();
    let e2 = A::knapscak_back_trace(val2, wei2, cap);
    println!("{:?}", ist.elapsed());
    assert_eq!(e, e2)
}

#[test]
fn e() {
   let mut a = i32::MAX;
   println!("{:?}",a + 1);
}
