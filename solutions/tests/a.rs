use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    f32::consts::PI,
    future, mem,
    net::{Ipv4Addr, Ipv6Addr},
    rc::Rc,
    str::FromStr,
    thread,
    time::{Duration, Instant},
    vec,
};

use solutions::{
    A, TreeNode,
    four::{self, Solution},
};

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
    let val = vec![
        1, 2, 3, 4, 5, 5, 6, 1, 2, 3, 4, 5, 5, 6, 1, 2, 3, 4, 5, 5, 6, 1, 5, 6, 1, 2, 3, 4,
    ];
    let val2 = val.clone();
    let wei = vec![
        2, 3, 4, 3, 4, 6, 6, 1, 2, 3, 4, 2, 3, 4, 5, 5, 6, 1, 5, 6, 1, 2, 3, 4, 5, 5, 6, 1,
    ];
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
    let mut map_1: HashMap<i32, i32> = HashMap::new();
    map_1.insert(1, 2);
    let mut map_2: HashMap<i32, i32> = HashMap::new();
    map_2.insert(1, 3);
    map_2.insert(2, 3);
    println!("{:?}", map_1 == map_2);

    fn show_all_t_sort(node_num: usize, map: Vec<(usize, usize)>) {
        let mut in_num = vec![0; node_num];
        let mut g = vec![vec![]; node_num];
        for (f, t) in map {
            in_num[t as usize] += 1;
            g[f as usize].push(t);
        }
        let mut start = VecDeque::new();
        for i in in_num.iter().enumerate() {
            if *i.1 == 0 {
                start.push_back(i.0);
            }
        }

        while let Some(node) = start.pop_front() {}
    }
}

#[test]
fn test_function2() {
    let nums = vec![1, -1, 5, 1, 4];
    let x = 1;
    let mut next = ((nums[x] + x as i32 + nums.len() as i32) % nums.len() as i32) as usize;
    println!("{:?}", (-1 + 1 + 5) % 5);
    println!("{:?}", next);
}

#[test]
fn adsad() {
    let a = vec![
        vec![-1, 83, -1, 46, -1, -1, -1, -1, 40, -1],
        vec![-1, 29, -1, -1, -1, 51, -1, 18, -1, -1],
        vec![-1, 35, 31, 51, -1, 6, -1, 40, -1, -1],
        vec![-1, -1, -1, 28, -1, 36, -1, -1, -1, -1],
        vec![-1, -1, -1, -1, 44, -1, -1, 84, -1, -1],
        vec![-1, -1, -1, 31, -1, 98, 27, 94, 74, -1],
        vec![4, -1, -1, 46, 3, 14, 7, -1, 84, 67],
        vec![-1, -1, -1, -1, 2, 72, -1, -1, 86, -1],
        vec![-1, 32, -1, -1, -1, -1, -1, -1, -1, 19],
        vec![-1, -1, -1, -1, -1, 72, 46, -1, 92, 6],
    ];
    let res = four::Solution::snakes_and_ladders(a);
    println!("{:?}", res);
}

#[test]
fn adsada() {
    Solution::conbination(vec![1, 2, 3, 4], 3);
}

#[test]
fn asdstr() {
    let s: i32 = vec![13, 11, 1, 8, 6, 7, 8, 8, 6, 7, 8, 9, 8].iter().sum();
    println!("{:?}", s);
    let mut s = vec![13, 11, 1, 8, 6, 7, 8, 8, 6, 7, 8, 9, 8];
    s.sort();
    println!("{:?}", s);
}

#[test]
fn adasdasd() {
    fn cals2(node: usize, max: usize) -> usize {
        let node = node * 10;
        if node > max {
            return 0;
        }
        let start = node;
        let end = max.min(start + 9);
        let mut res = end - start + 1;
        for i in start..=end {
            let s = cals2(i, max);
            if s == 0 {
                break;
            }
            res += s;
        }
        res
    }
    fn cals(node: usize, max: usize) -> usize {
        let mut step2 = 0;
        let mut f = node * 10;
        let mut l = node * 10 + 9;
        while f <= max {
            step2 += l.min(max) - f + 1;
            f *= 10;
            l = l * 10 + 9;
        }
        step2
    }
    let a = 95707474;
    let mut d = Duration::new(0, 0);
    let mut e = Duration::new(0, 0);
    for i in 1..a {
        let x = Instant::now();
        let aa = cals(i, a);
        d += x.elapsed();
        let x = Instant::now();
        let bb = cals2(i, a);
        e += x.elapsed();
        assert_eq!(aa, bb);
    }
    println!("{:?} my{:?}", d, e);
    return;
    let i = Instant::now();
    // return;
    fn search(node: usize, k: usize, n: usize, mut current: usize) -> usize {
        for i in 0..10 {
            current += 1;
            let node = i + node;
            if current == k {
                return node;
            }
            let next = cals(node, n);
            if current + next < k {
                println!("--{:?} {} {}", current, next, node);
                current += next;
            } else if current + next > k {
                println!("{:?} {}", current, next);

                return search(node * 10, k, n, current);
            } else {
                return node;
            }
        }
        0
    }
    let k = 424238336;
    let n = 957747794;
    let mut current = 0;
    for i in 1..=9 {
        current += 1;
        let next = cals(i, n);
        if current == k {
            println!("ans2 = {:?}", i);

            break;
        }
        if current + next < k {
            current += next;
        } else if current + next >= k {
            let res = search(i * 10, k, n, current);
            println!("ans = {:?}", res);
            break;
        } else {
            println!("ans1 = {:?}", i);
            break;
        }
    }
    println!("{:?}", i.elapsed());
    pub fn find_max_average(nums: Vec<i32>, k: i32) -> f64 {
        let mut res = 0.0_f64;
        for i in 0..nums.len() - k as usize {
            println!("{:?}", (&nums[i..i + k as usize]).iter().sum::<i32>());
            res = res.max((&nums[i..i + k as usize]).iter().sum::<i32>() as f64 / k as f64);
        }
        res
    }
}

#[test]
fn fuckckck() {
    fn max_difference(s: String, k: i32) -> i32 {
        let k = k as usize;
        let mut map = vec![0; 5];
        let s = s.as_bytes();
        for i in 0..k - 1 {
            map[s[i] as usize - 48] += 1;
        }
        fn max_dif(map: &[i32]) -> ((i32, usize), (i32, usize), i32) {
            let mut x = 0;
            for i in 0..map.len() {
                if map[i] % 2 == 0 {
                    x &= !(1 << i)
                } else {
                    x |= 1 << i
                }
            }
            if x == 0 || x == 31 {
                // all even, ||
                return ((0, 0), (0, 0), x);
            }

            let mut max_odd = i32::MIN;
            let mut max_odd_idx = 0;
            let mut min_even = i32::MAX;
            let mut min_even_idx = 0;
            for (idx, v) in map.iter().enumerate() {
                if *v != 0 {
                    if *v % 2 == 0 {
                        if min_even > *v {
                            min_even = *v;
                            min_even_idx = idx;
                        }
                    } else {
                        if max_odd < *v {
                            max_odd = *v;
                            max_odd_idx = idx;
                        }
                    }
                }
            }
            let mut x = 0;
            for i in 0..map.len() {
                if map[i] % 2 == 0 {
                    x &= !(1 << i)
                } else {
                    x |= 1 << i
                }
            }
            ((max_odd, max_odd_idx), (min_even, min_even_idx), x)
        }
        fn slide(len: usize, mut map: &mut Vec<i32>, s: &[u8]) -> i32 {
            let mut max = max_dif(&map);
            // println!("----- {:?}  {:?}",max,map);

            for i in len..s.len() {
                map[s[i - len] as usize - 48] -= 1;
                if (max.2 >> (s[i - len] as usize - 48)) & 1 == 1 {
                    max.2 &= !(1 << (s[i - len] as usize - 48))
                } else {
                    max.2 |= 1 << (s[i - len] as usize - 48)
                }
                map[s[i] as usize - 48] += 1;
                if (max.2 >> (s[i] as usize - 48)) & 1 == 1 {
                    max.2 &= !(1 << (s[i] as usize - 48))
                } else {
                    max.2 |= 1 << (s[i] as usize - 48)
                }
                if max.2 != 0 && max.2 != 31 {
                    let max2 = max.max(max_dif(&map));
                    if max2.0.0 - max2.1.0 > max.0.0 - max.1.0 {
                        max = max2
                    }
                }
                // println!("{:?}  {:?}",max,map);
            }
            max.0.0 - max.1.0
        }
        let mut res = i32::MIN;
        let mut temp = vec![0; 5];
        for i in k - 1..s.len() {
            map[s[i] as usize - 48] += 1;
            for i in 0..map.len() {
                temp[i] = map[i]
            }
            res = res.max(slide(i + 1, &mut map, &s));
            for i in 0..map.len() {
                map[i] = temp[i]
            }
        }
        println!("{:?}", res);
        res
    }
    let s = include_str!("../../input.txt");
    let s = s.to_string();
    let i = Instant::now();
    let res = max_difference(s, 2880);
    println!("{:?}", i.elapsed());
    println!("re{:?}", res);
}

#[test]
fn aaaa() {
    fn max_dif(map: &[i32]) -> ((i32, usize), (i32, usize), i32) {
        let mut x = 0;
        for i in 0..map.len() {
            if map[i] % 2 == 0 {
                x &= !(1 << i)
            } else {
                x |= 1 << i
            }
        }
        if x == 0 || x == 31 {
            // all even, ||
            return ((0, 0), (0, 0), x);
        }
        let mut max_odd = i32::MIN;
        let mut max_odd_idx = 0;
        let mut min_even = i32::MAX;
        let mut min_even_idx = 0;
        for (idx, v) in map.iter().enumerate() {
            if *v != 0 {
                if *v % 2 == 0 {
                    if min_even > *v {
                        min_even = *v;
                        min_even_idx = idx;
                    }
                } else {
                    if max_odd < *v {
                        max_odd = *v;
                        max_odd_idx = idx;
                    }
                }
            }
        }

        ((max_odd, max_odd_idx), (min_even, min_even_idx), x)
    }
    fn slide(len: usize, mut map: &mut Vec<i32>, s: &[u8]) -> i32 {
        let mut max = max_dif(&map);
        // println!("----- {:?}  {:?}",max,map);

        for i in len..s.len() {
            map[s[i - len] as usize - 48] -= 1;
            if (max.2 >> (s[i - len] as usize - 48)) & 1 == 1 {
                max.2 &= !(1 << (s[i - len] as usize - 48))
            } else {
                max.2 |= 1 << (s[i - len] as usize - 48)
            }
            map[s[i] as usize - 48] += 1;
            if (max.2 >> (s[i] as usize - 48)) & 1 == 1 {
                max.2 &= !(1 << (s[i] as usize - 48))
            } else {
                max.2 |= 1 << (s[i] as usize - 48)
            }
            if max.2 != 0 && max.2 != 31 {
                let max2 = max.max(max_dif(&map));
                if max2.0.0 - max2.1.0 > max.0.0 - max.1.0 {
                    max = max2
                }
            }
            // println!("{:?}  {:?}",max,map);
        }
        max.0.0 - max.1.0
    }
    let res = max_dif(&[2, 2, 1, 4, 6]);
    println!("{:?}, {:b}", res, res.2);
}

#[test]
fn life_time() {
    struct A<'a> {
        orign: Vec<String>,
        map: HashMap<&'a str, usize>,
    }
    impl<'a> A<'a> {
        fn new() -> A<'a> {
            Self {
                orign: vec![],
                map: HashMap::new(),
            }
        }
        fn insert(&'a mut self) {
            let s = String::new();

            self.orign.push(s);
            let k = self.orign.last().unwrap();
            self.map.insert(k, 0);
        }
    }
    let mut a: A<'_> = A::new();
    a.insert();
}

fn easdasd() {
}
