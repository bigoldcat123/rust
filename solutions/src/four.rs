pub struct Solution {}

impl Solution {
    //401
    pub fn read_binary_watch(turned_on: i32) -> Vec<String> {
        let mut res = vec![];
        if turned_on > 8 {
            return res;
        }
        fn search(
            steps: &[i32],
            current: i32,
            start: usize,
            limit: usize,
            current_num: usize,
            max: i32,
            res: &mut Vec<String>,
        ) {
            if limit == 0 {
                res.push("0".to_string());
                return;
            }
            if current_num >= limit && current <= max {
                res.push(current.to_string());
                return;
            }
            if start >= steps.len() {
                return;
            }
            for i in start..steps.len() {
                search(
                    steps,
                    current + steps[i],
                    i + 1,
                    limit,
                    current_num + 1,
                    max,
                    res,
                );
            }
        }
        let step_hour = [8, 4, 2, 1];
        let step_min = [32, 16, 8, 4, 2, 1];
        for i in 0..=3 {
            let min_num = turned_on - i;
            if min_num >= 0 {
                let mut hour = vec![];
                let mut min = vec![];
                search(&step_hour, 0, 0, i as usize, 0, 11, &mut hour);
                search(&step_min, 0, 0, min_num as usize, 0, 59, &mut min);
                for h in hour {
                    for m in min.iter() {
                        if m.len() == 1 {
                            res.push(format!("{}:0{}", h, m));
                        } else {
                            res.push(format!("{}:{}", h, m));
                        }
                    }
                }
            }
        }
        res
    }

    //402
    pub fn remove_kdigits(num: String, k: i32) -> String {
        let k = k as usize;
        if k == num.len() {
            return "0".to_string();
        }
        let mut res = String::new();
        let mut opt_times = 0;
        let a = &num[..];
        let num = num.as_bytes();
        let mut start = num.len();

        let mut pre_take_it = true;

        for i in 0..num.len() {
            if k - opt_times >= num.len() - i {
                break;
            }
            if opt_times >= k {
                start = i;
                break;
            }
            let mut take_it = true;
            if i > 0 {
                if num[i] == num[i - 1] && pre_take_it {
                    if num[i + k - opt_times] < num[i] {
                        take_it = false;
                        pre_take_it = false;
                    }
                } else {
                    for j in i + 1..=(i + k - opt_times) {
                        if num[j] < num[i] {
                            take_it = false;
                            pre_take_it = false;
                            break;
                        }
                    }
                }
            } else {
                for j in i + 1..=(i + k - opt_times).min(num.len() - 1) {
                    if num[j] < num[i] {
                        take_it = false;
                        pre_take_it = false;
                        break;
                    }
                }
            }
            if take_it {
                res.push(num[i] as char);
                pre_take_it = true;
            } else {
                opt_times += 1;
            }
        }
        for i in start..num.len() {
            let n = num[i] as char;
            res.push(n);
        }
        // res.push_str(&a[start..]);
        let mut start = res.len();
        for i in 0..res.as_bytes().len() {
            if res.as_bytes()[i] != 48 {
                start = i;
                break;
            }
        }
        if start == res.len() {
            return "0".to_string();
        }
        format!("{}", &res[start..])
    }

    //405
    pub fn to_hex(num: i32) -> String {
        use std::collections::VecDeque;
        let origin = num;
        let mut num = num.abs();
        let mut b_vec = VecDeque::new();
        let mut res = String::new();
        while num > 0 {
            let x = num % 2;
            b_vec.push_front(x);
            num /= 2;
        }
        while b_vec.len() % 4 != 0 {
            b_vec.push_front(0);
        }
        if origin < 0 {
            let left = 32 - b_vec.len();
            for i in 1..left {
                b_vec.push_front(0);
            }
            b_vec.push_front(1);
            for i in 1..32 {
                if b_vec[i] == 0 {
                    b_vec[i] = 1;
                } else {
                    b_vec[i] = 0;
                }
            }
            if *b_vec.iter().last().unwrap() == 1 {
                for i in (0..32).rev() {
                    if b_vec[i] == 0 {
                        b_vec[i] = 1;
                        break;
                    } else {
                        b_vec[i] = 0;
                    }
                }
            } else {
                b_vec[31] = 1;
            }
        }
        b_vec.make_contiguous();
        let (b_vec, _) = b_vec.as_slices();
        for i in (0..b_vec.len()).step_by(4) {
            match &b_vec[i..i + 4] {
                [0, 0, 0, 0] => {
                    res.push('0');
                }
                [0, 0, 0, 1] => {
                    res.push('1');
                }
                [0, 0, 1, 0] => {
                    res.push('2');
                }
                [0, 0, 1, 1] => {
                    res.push('3');
                }
                [0, 1, 0, 0] => {
                    res.push('4');
                }
                [0, 1, 0, 1] => {
                    res.push('5');
                }
                [0, 1, 1, 0] => {
                    res.push('6');
                }
                [0, 1, 1, 1] => {
                    res.push('7');
                }
                [1, 0, 0, 0] => {
                    res.push('8');
                }
                [1, 0, 0, 1] => {
                    res.push('9');
                }
                [1, 0, 1, 0] => {
                    res.push('a');
                }
                [1, 0, 1, 1] => {
                    res.push('b');
                }
                [1, 1, 0, 0] => {
                    res.push('c');
                }
                [1, 1, 0, 1] => {
                    res.push('d');
                }
                [1, 1, 1, 0] => {
                    res.push('e');
                }
                [1, 1, 1, 1] => {
                    res.push('f');
                }
                _ => {}
            }
        }
        res
    }
}
