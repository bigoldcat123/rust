#![allow(dead_code, unused)]
use core::{num, time};
use std::{cell::RefCell, collections::HashSet, i32, rc::Rc, str::FromStr, usize, vec};

use log::log_enabled;
use rand::rand_core::le;

use crate::TreeNode;

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
            for _ in 1..left {
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
    //406
    pub fn reconstruct_queue(people: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut people = people;
        people.sort_by(|p1, p2| {
            if p1[0] != p2[0] {
                p1[0].cmp(&p2[0])
            } else {
                p2[1].cmp(&p1[1])
            }
        });
        let mut res = vec![vec![]; people.len()];

        for p in people {
            let mut space = p[1] + 1;
            for i in 0..res.len() {
                if res[i].is_empty() {
                    space -= 1;
                    if space == 0 {
                        res[i] = p;
                        break;
                    }
                }
            }
        }
        res
    }
    //409
    pub fn longest_palindrome(s: String) -> i32 {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let s = s.as_bytes();
        for b in s {
            if let Some(v) = map.get_mut(b) {
                *v += 1;
            } else {
                map.insert(*b, 1);
            }
        }
        let mut res = 0;
        let mut max_odd = 0;
        for (_, v) in map {
            if v % 2 != 0 {
                if v > max_odd {
                    res += max_odd - 1;
                    max_odd = v;
                } else {
                    res += v - 1;
                }
            } else {
                res += v;
            }
        }
        res + max_odd
    }
    //413
    pub fn number_of_arithmetic_slices(nums: Vec<i32>) -> i32 {
        let mut dp = vec![vec![true; nums.len()]; nums.len()];
        let mut res = 0;
        for i in 2..nums.len() {
            for j in 0..nums.len() - i {
                dp[j][j + i] = if dp[j][j + i - 1]
                    && nums[j + i] - nums[j + i - 1] == nums[j + i - 1] - nums[j + i - 2]
                {
                    res += 1;
                    true
                } else {
                    false
                }
            }
        }
        res
    }
    //414
    pub fn third_max(nums: Vec<i32>) -> i32 {
        let mut nums = nums;

        nums.sort();
        nums.dedup();
        if nums.len() < 3 {
            *nums.last().unwrap()
        } else {
            nums[nums.len() - 3]
        }
    }

    //415
    pub fn add_strings(num1: String, num2: String) -> String {
        let num1 = num1.as_bytes();
        let num2 = num2.as_bytes();
        let mut res = vec![];

        let mut idx_1 = num1.len() - 1;
        let mut idx_2 = num2.len() - 1;
        let mut carry = 0;

        while idx_1 < num1.len() || idx_2 < num2.len() {
            let mut left = 0;
            let mut right = 0;

            if idx_1 < num1.len() {
                left = num1[idx_1] - 48;
            }
            if idx_2 < num2.len() {
                right = num2[idx_2] - 48;
            }
            let r = right + left + carry;
            res.push(r % 10 + 48);
            carry = r / 10;
            idx_1 -= 1;
            idx_2 -= 1;
        }
        if carry != 0 {
            res.push(carry);
        };
        res.reverse();
        String::from_utf8(res).unwrap()
    }
    //29
    pub fn divide(mut dividend: i32, mut divisor: i32) -> i32 {
        if dividend == i32::MIN {
            if divisor == -1 {
                return i32::MAX;
            }
        }
        if divisor == i32::MAX {
            if dividend == i32::MAX {
                return 1;
            } else {
                return 0;
            }
        }
        let mut obs = false;
        if dividend > 0 {
            dividend = -dividend;
            obs = !obs;
        }
        if divisor > 0 {
            divisor = -divisor;
            obs = !obs;
        }
        let mut left = 1;
        let mut right = i32::MAX;

        let mut res = 1;
        while left < right {
            let mid = (left + right) / 2;
            if mid * divisor >= dividend {
                res = mid;
                left = mid + 1;
            } else if mid * divisor < dividend {
                right = mid - 1;
            }
        }

        if obs { -res } else { res }
    }
    //416
    pub fn can_partition(nums: Vec<i32>) -> bool {
        let sum = nums.iter().sum::<i32>();
        if sum % 2 != 0 {
            return false;
        }
        let amount = sum as usize / 2;
        let mut dp = vec![vec![true; nums.len()]; amount + 1];
        for i in 1..amount + 1 {
            dp[i][0] = nums[0] == i as i32;
        }
        for i in 1..amount + 1 {
            for j in 0..nums.len() {
                if dp[i][j - 1] {
                    dp[i][j] = true;
                } else {
                    if nums[j] as usize > i {
                        dp[i][j] = false;
                    } else {
                        dp[i][j] = dp[i - nums[j] as usize][j - 1];
                    }
                }
            }
        }
        dp.last().unwrap().contains(&true)
    }
    //417
    pub fn pacific_atlantic(heights: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        #[derive(Clone, Copy, Debug)]
        struct Node {
            visited_p: bool,
            visited_a: bool,
            paciific: bool,
            atlantic: bool,
        }
        impl Node {
            fn new() -> Self {
                Self {
                    visited_p: false,
                    visited_a: false,
                    paciific: false,
                    atlantic: false,
                }
            }
        }
        let mut h = vec![vec![Node::new(); heights[0].len()]; heights.len()];
        let mut r = vec![];

        fn search(
            c: (usize, usize),
            from: (usize, usize),
            t: i32,
            heights: &Vec<Vec<i32>>,
            h: &mut Vec<Vec<Node>>,
        ) -> bool {
            if (c.0 == 0 || c.1 == 0) && t == 1 {
                return true;
            }
            if (c.0 == heights.len() - 1 || c.1 == heights[0].len() - 1) && t == 2 {
                return true;
            }
            // println!("search {:?}",c);
            let node = &mut h[c.0][c.1];
            if t == 1 && node.visited_p {
                node.paciific
            } else if t == 2 && node.visited_a {
                node.atlantic
            } else {
                let current_height = heights[c.0][c.1];

                if c.1 != 0 {
                    let left = (c.0, c.1 - 1);
                    if c.1 != 0 && heights[left.0][left.1] <= current_height {
                        let current_node = &mut h[c.0][c.1];

                        if t == 1 {
                            current_node.visited_p = true;
                        } else {
                            current_node.visited_a = true;
                        }

                        let r = search(left, c, t, heights, h);
                        let left_node = &mut h[left.0][left.1];
                        if t == 1 {
                            left_node.paciific = r;
                            left_node.visited_p = true;
                        } else {
                            left_node.atlantic = r;
                            left_node.visited_a = true;
                        }
                        if r {
                            return true;
                        }
                    }
                }
                if c.1 != heights[0].len() - 1 {
                    let right = (c.0, c.1 + 1);

                    if heights[right.0][right.1] <= current_height {
                        let current_node = &mut h[c.0][c.1];
                        if t == 1 {
                            current_node.visited_p = true;
                        } else {
                            current_node.visited_a = true;
                        }
                        let r = search(right, c, t, heights, h);
                        let right_node = &mut h[right.0][right.1];
                        if t == 1 {
                            right_node.paciific = r;
                            right_node.visited_p = true;
                        } else {
                            right_node.atlantic = r;
                            right_node.visited_a = true;
                        }
                        if r {
                            return true;
                        }
                    }
                }
                if c.0 != 0 {
                    let top = (c.0 - 1, c.1);

                    if heights[top.0][top.1] <= current_height {
                        let current_node = &mut h[c.0][c.1];
                        if t == 1 {
                            current_node.visited_p = true;
                        } else {
                            current_node.visited_a = true;
                        }
                        let r = search(top, c, t, heights, h);
                        let top_node = &mut h[top.0][top.1];
                        if t == 1 {
                            top_node.paciific = r;
                            top_node.visited_p = true;
                        } else {
                            top_node.atlantic = r;
                            top_node.visited_a = true;
                        }
                        if r {
                            return true;
                        }
                    }
                }
                if c.0 != heights.len() - 1 {
                    let down = (c.0 + 1, c.1);

                    if heights[down.0][down.1] <= current_height {
                        let current_node = &mut h[c.0][c.1];
                        if t == 1 {
                            current_node.visited_p = true;
                        } else {
                            current_node.visited_a = true;
                        }
                        let r = search(down, c, t, heights, h);
                        let down_node = &mut h[down.0][down.1];
                        if t == 1 {
                            down_node.paciific = r;
                            down_node.visited_p = true;
                        } else {
                            down_node.atlantic = r;
                            down_node.visited_a = true;
                        }
                        if r {
                            return true;
                        }
                    }
                }
                false
            }
        }
        for i in 0..heights.len() {
            for j in 0..heights[0].len() {
                h[i][j].visited_p = false;
                h[i][j].visited_a = false;
                let res = search((i, j), (i, j), 1, &heights, &mut h);
                h[i][j].paciific = res;
                h[i][j].visited_p = true;

                let res = search((i, j), (i, j), 2, &heights, &mut h);
                h[i][j].atlantic = res;
                h[i][j].visited_a = true;
                if i == 35 {
                    println!("{:?} {:?}", h[i][j], heights[i][j]);
                }
                if h[i][j].paciific && h[i][j].atlantic {
                    r.push(vec![i as i32, j as i32]);
                }
            }
        }
        r
    }

    //419
    pub fn count_battleships(board: Vec<Vec<char>>) -> i32 {
        let mut res = 0;
        let mut board_mark = vec![vec![false; board[0].len()]; board.len()];
        fn mark(
            board: &Vec<Vec<char>>,
            board_mark: &mut Vec<Vec<bool>>,
            dir_left: bool,
            mut c: (usize, usize),
        ) {
            if dir_left {
                while board[c.0][c.1] != '.' {
                    board_mark[c.0][c.1] = true;
                    c.1 += 1;
                    if c.1 == board[0].len() {
                        break;
                    }
                }
            } else {
                while board[c.0][c.1] != '.' {
                    board_mark[c.0][c.1] = true;
                    c.0 += 1;
                    if c.0 == board.len() {
                        break;
                    }
                }
            }
        }
        for i in 0..board.len() {
            for j in 0..board.len() {
                if board[i][j] == 'X' && !board_mark[i][j] {
                    res += 1;
                    mark(&board, &mut board_mark, true, (i, j));
                    mark(&board, &mut board_mark, false, (i, j));
                }
            }
        }

        res
    }
    //421
    pub fn find_maximum_xor(nums: Vec<i32>) -> i32 {
        use std::collections::HashSet;
        let mut res = 0;
        for i in (0..=31).rev() {
            let mut set = HashSet::new();
            for n in nums.iter() {
                set.insert(*n >> i);
            }

            let next = (res << 1) + 1;

            let mut found = false;

            for n in nums.iter() {
                if set.contains(&((*n >> i) ^ next)) {
                    found = true;
                    break;
                }
            }
            if found {
                res = next;
            } else {
                res = next - 1;
            }
        }

        res
    }

    //423
    pub fn original_digits(s: String) -> String {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let mut res = vec![0; 10];
        for i in s.into_bytes() {
            if let Some(v) = map.get_mut(&i) {
                *v += 1;
            } else {
                map.insert(i, 1);
            }
        }
        res[0] = map.remove(&b'z').unwrap_or(0);
        res[2] = map.remove(&b'w').unwrap_or(0);
        res[4] = map.remove(&b'u').unwrap_or(0);
        res[6] = map.remove(&b'x').unwrap_or(0);
        res[8] = map.remove(&b'g').unwrap_or(0);

        res[3] = map.remove(&b'h').unwrap_or(0) - res[8];
        res[5] = map.remove(&b'f').unwrap_or(0) - res[4];
        res[7] = map.remove(&b's').unwrap_or(0) - res[6];

        res[1] = map.remove(&b'o').unwrap_or(0) - res[0] - res[2] - res[4];
        res[9] = map.remove(&b'i').unwrap_or(0) - res[6] - res[5] - res[8];
        let mut r = String::new();
        for (idx, v) in res.into_iter().enumerate() {
            for _ in 0..v {
                r.push((idx as u8 + 48) as char);
            }
        }
        r
    }

    //424
    pub fn character_replacement(s: String, k: i32) -> i32 {
        let mut res = 0;

        let k = k as usize;
        let mut left = 0;
        let mut right = 0;

        let s = s.as_bytes();
        let mut max_count = 0;
        let mut map = [0; 26];
        while right < s.len() {
            let idx = (s[right] - b'A') as usize;
            map[idx] += 1;
            max_count = max_count.max(map[idx]);
            right += 1;

            if right - left > max_count + k {
                let idx = (s[left] - b'A') as usize;
                map[idx] -= 1;

                left += 1;
            }

            res = res.max(right - left);
        }

        res as i32
    }
    //2942
    pub fn find_words_containing(words: Vec<String>, x: char) -> Vec<i32> {
        let mut res = vec![];
        for (i, word) in words.into_iter().enumerate() {
            if word.contains(x) {
                res.push(i as i32);
            }
        }
        res
    }

    //449
    fn solution_449() {
        struct Codec {}

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl Codec {
            fn new() -> Self {
                Self {}
            }

            fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
                let mut pre_order = vec![];
                let mut mid_order = vec![];
                fn build_pre_order(
                    root: Option<Rc<RefCell<TreeNode>>>,
                    pre_order: &mut Vec<String>,
                ) {
                    if let Some(root) = root {
                        let root = root.borrow();
                        pre_order.push(root.val.to_string());
                        build_pre_order(root.left.clone(), pre_order);
                        build_pre_order(root.right.clone(), pre_order);
                    }
                }
                fn build_mid_order(
                    root: Option<Rc<RefCell<TreeNode>>>,
                    mid_order: &mut Vec<String>,
                ) {
                    if let Some(root) = root {
                        let root = root.borrow();
                        build_mid_order(root.left.clone(), mid_order);
                        mid_order.push(root.val.to_string());
                        build_mid_order(root.right.clone(), mid_order);
                    }
                }
                build_mid_order(root.clone(), &mut mid_order);
                build_pre_order(root, &mut pre_order);
                let pre_order = pre_order.join(",");
                let mid_order = mid_order.join(",");
                format!("{}#{}", pre_order, mid_order)
            }

            fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
                let (pre, mid) = data.split_once("#").unwrap();
                let pre_order = pre
                    .split(",")
                    .map(|x| x.parse::<i32>().unwrap())
                    .collect::<Vec<i32>>();
                let mid_order = mid
                    .split(",")
                    .map(|x| x.parse::<i32>().unwrap())
                    .collect::<Vec<i32>>();

                fn build_tree(
                    pre_order: &[i32],
                    mid_order: &[i32],
                ) -> Option<Rc<RefCell<TreeNode>>> {
                    if pre_order.len() != 0 {
                        let mut root = TreeNode::new(pre_order[0]);
                        let idx = mid_order.binary_search(&pre_order[0]).unwrap();

                        root.left = build_tree(&pre_order[1..idx + 1], &mid_order[..idx]);
                        root.right = build_tree(&pre_order[idx + 1..], &mid_order[idx + 1..]);
                        Some(Rc::new(RefCell::new(root)))
                    } else {
                        None
                    }
                }
                build_tree(&pre_order, &mid_order)
            }
        }
    }
    //450
    pub fn delete_node(
        root: Option<Rc<RefCell<TreeNode>>>,
        key: i32,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        fn dfs_tree(root: &mut Option<Rc<RefCell<TreeNode>>>, k: i32) {
            let inner = root.take();
            if let Some(inner) = inner {
                if inner.borrow().val == k {
                    let left = inner.borrow_mut().left.take();
                    let right = inner.borrow_mut().right.take();
                    match (left, right) {
                        (None, None) => {
                            *root = None;
                        }
                        (Some(left), None) => {
                            *root = Some(left);
                        }
                        (None, Some(right)) => {
                            *root = Some(right);
                        }

                        (Some(left), Some(right)) => {
                            let mut l = left.clone();
                            loop {
                                if l.borrow_mut().right.is_none() {
                                    l.borrow_mut().right = Some(right);
                                    break;
                                } else {
                                    let e = l.borrow_mut().right.clone();
                                    l = e.unwrap();
                                }
                            }
                            *root = Some(left);
                        }
                    }
                } else {
                    if inner.borrow().val > k {
                        dfs_tree(&mut inner.borrow_mut().left, k);
                    } else {
                        dfs_tree(&mut inner.borrow_mut().right, k);
                    }
                    *root = Some(inner);
                }
            }
        }
        let mut root = root;
        dfs_tree(&mut root, key);
        root
    }
    //2131
    pub fn longest_palindrome_2131(words: Vec<String>) -> i32 {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let mut res = 0;
        let mut middle = false;
        for w in words.iter() {
            if let Some(v) = map.get_mut(w.as_bytes()) {
                *v += 1;
            } else {
                map.insert(w.as_bytes(), 1);
            }
        }
        for word in words.iter() {
            let word = word.as_bytes();
            if let Some(v) = map.get_mut(&[word[1], word[0]] as &[u8]) {
                if *v > 1 && word[0] != word[1] {
                    *v -= 1;
                    res += 4;
                } else if *v == 1 && word[0] == word[1] {
                    middle = true;
                } else if *v >= 2 && word[0] == word[1] {
                    *v -= 1;
                    res += 4;
                }
            }
            *map.get_mut(word).unwrap() -= 1;
        }
        if middle {
            res += 2;
        }
        res
    }

    //451
    pub fn frequency_sort(s: String) -> String {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let mut res = String::new();
        for i in s.as_bytes() {
            if let Some(v) = map.get_mut(i) {
                *v += 1;
            } else {
                map.insert(*i, 1);
            }
        }
        let mut arr: Vec<(u8, i32)> = map.into_iter().map(|x| x).collect();
        arr.sort_by(|x, y| y.1.cmp(&x.1));
        for w in arr {
            res.push_str(&format!("{}", w.0 as char).repeat(w.1 as usize));
        }
        res
    }

    //452
    pub fn find_min_arrow_shots(points: Vec<Vec<i32>>) -> i32 {
        let mut points = points;
        points.sort_by(|x, y| x[0].cmp(&y[0]));
        let mut res = 0;
        let mut left = 0;
        let mut right = 1;
        let mut max = points[0][1];
        while right < points.len() {
            if points[right][0] <= max {
                max = max.max(points[right][1]);
            } else {
                res += 1;
                left = right;
                right = right + 1;
                max = points[left][1];
            }
        }
        res
    }
    fn asd() {}
    //447
    pub fn number_of_boomerangs(points: Vec<Vec<i32>>) -> i32 {
        use std::collections::HashMap;
        let mut res = 0;
        for i in 0..points.len() {
            let left = (points[i][0], points[i][1]);
            let mut map = HashMap::new();
            for j in 0..points.len() {
                let right = (points[j][0], points[j][1]);
                let dist = (right.0 - left.0) * (right.0 - left.0)
                    + (right.1 - left.1) * (right.1 - left.1);
                if let Some(v) = map.get_mut(&dist) {
                    *v += 1;
                } else {
                    map.insert(dist, 1);
                }
            }
            for (k, v) in map {
                res += v * (v - 1)
            }
        }
        res
    }
    //448
    pub fn find_disappeared_numbers(nums: Vec<i32>) -> Vec<i32> {
        let mut nums = nums;
        let len = nums.len();
        for i in 0..len {
            while nums[i] as usize != i + 1 {
                let target = nums[i] as usize - 1;
                if nums[target] == nums[i] {
                    break;
                }
                let t = nums[i];
                nums[i] = nums[target];
                nums[target] = t;
            }
        }
        let mut res = vec![];
        for i in 0..len {
            if nums[i] as usize != i + 1 {
                res.push(i as i32 + 1);
            }
        }

        res
    }

    //1857
    /// dp[u][c], n as the end node, the max colors of c;
    /// dp[u][c] = max(dp[v][c]) -- v as the pre_nodes to u
    ///
    pub fn largest_path_value(colors: String, edges: Vec<Vec<i32>>) -> i32 {
        use std::collections::VecDeque;
        let colors = colors.as_bytes();
        let node_num = colors.len();
        let mut in_edge = vec![0; node_num];
        let mut g = vec![vec![]; node_num];
        for edge in edges {
            in_edge[edge[1] as usize] += 1;
            g[edge[0] as usize].push(edge[1] as usize);
        }
        let mut start_nodes = VecDeque::new();
        let mut dp = vec![vec![0; 26]; node_num];
        for (idx, in_num) in in_edge.iter().enumerate() {
            if *in_num == 0 {
                start_nodes.push_back(idx);
            }
        }
        let mut found = 0;

        while let Some(next) = start_nodes.pop_front() {
            dp[next][colors[next] as usize - 97] += 1;
            found += 1;
            for n in g[next].iter() {
                in_edge[*n] -= 1;

                for i in 0..26 {
                    dp[*n][i] = dp[*n][i].max(dp[next][i]);
                }

                if in_edge[*n] == 0 {
                    start_nodes.push_back(*n);
                }
            }
        }
        if found != node_num {
            return -1;
        }
        let mut res = 0;

        for e in dp {
            res = res.max(*e.iter().max().unwrap());
        }

        res
    }

    //2894
    pub fn difference_of_sums(n: i32, m: i32) -> i32 {
        let mut num1 = 0;
        let mut num2 = 0;
        for i in 1..=n {
            if i % m != 0 {
                num1 += i;
            } else {
                num2 += i;
            }
        }
        num1 - num2
    }
    //454
    pub fn four_sum_count(
        nums1: Vec<i32>,
        nums2: Vec<i32>,
        nums3: Vec<i32>,
        nums4: Vec<i32>,
    ) -> i32 {
        let mut res = 0;
        // let mut num_1_2 = vec![];
        // let mut num_3_4 = vec![];
        // for i in nums1 {
        //     for j in &nums2 {
        //         num_1_2.push(i + *j);
        //     }
        // }

        // for i in nums3 {
        //     for j in &nums4 {
        //         num_3_4.push(i + *j);
        //     }
        // }

        // let mut neg_num_1_2 = num_1_2.iter().filter(|x| **x < 0).collect::<Vec<&i32>>();
        // let mut post_num_3_4 = num_3_4.iter().filter(|x| **x > 0).collect::<Vec<&i32>>();
        // post_num_3_4.sort_by(|a, b| b.cmp(a));
        // neg_num_1_2.sort();

        // let mut i = 0;
        // let mut j = 0;

        // loop {
        //     if i >= post_num_3_4.len() || j >= neg_num_1_2.len() {
        //         break;
        //     }
        //     let post_number = post_num_3_4[i];
        //     let neg_number = neg_num_1_2[j];
        //     if *post_number == -*neg_number {

        //         let mut x = 0;
        //         while i < post_num_3_4.len() && *post_num_3_4[i] == -*neg_number {
        //             x += 1;
        //             i += 1;
        //         }
        //         let mut y = 0;
        //         while j < neg_num_1_2.len() && *post_number == -*neg_num_1_2[j] {
        //             y += 1;
        //             j += 1;
        //         }
        //         res += y * x;
        //         //10 > --9
        //     } else if *post_number > -*neg_number {
        //         i += 1;
        //     } else {
        //         //10 < --11
        //         j += 1
        //     }
        // }

        // let mut neg_num_1_2 = num_3_4.iter().filter(|x| **x < 0).collect::<Vec<&i32>>();
        // let mut post_num_3_4 = num_1_2.iter().filter(|x| **x > 0).collect::<Vec<&i32>>();
        // post_num_3_4.sort_by(|a, b| b.cmp(a));
        // neg_num_1_2.sort();

        // let mut i = 0;
        // let mut j = 0;

        // loop {
        //     if i >= post_num_3_4.len() || j >= neg_num_1_2.len() {
        //         break;
        //     }
        //     let post_number = post_num_3_4[i];
        //     let neg_number = neg_num_1_2[j];
        //     if *post_number == -*neg_number {
        //         let mut x = 0;
        //         while i < post_num_3_4.len() && *post_num_3_4[i] == -*neg_number {
        //             x += 1;
        //             i += 1;
        //         }
        //         let mut y = 0;
        //         while j < neg_num_1_2.len() && *post_number == -*neg_num_1_2[j] {
        //             y += 1;
        //             j += 1;
        //         }
        //         res += y * x;
        //         //10 > --9
        //     } else if *post_number > -*neg_number {
        //         i += 1;
        //     } else {
        //         //10 < --11
        //         j += 1
        //     }
        // }

        // let mut neg_num_1_2 = num_3_4.iter().filter(|x| **x == 0).collect::<Vec<&i32>>();
        // let mut post_num_3_4 = num_1_2.iter().filter(|x| **x == 0).collect::<Vec<&i32>>();
        // res += neg_num_1_2.len() as i32 * post_num_3_4.len() as i32;

        use std::collections::HashMap;
        let mut res = 0;
        let mut map = HashMap::new();
        for i in nums1.iter() {
            for j in nums2.iter() {
                if let Some(v) = map.get_mut(&(*j + *i)) {
                    *v += 1;
                } else {
                    map.insert(*j + *i, 1);
                }
            }
        }

        for i in nums3.iter() {
            for j in nums4.iter() {
                let key = *j * -*j;
                if let Some(v) = map.get_mut(&key) {
                    if *v >= 1 {
                        res += 1;
                        *v -= 1;
                    }
                }
            }
        }
        res
    }

    //455
    pub fn find_content_children(g: Vec<i32>, s: Vec<i32>) -> i32 {
        let mut g = g;
        let mut s = s;
        g.sort();
        s.sort();
        let mut i = 0;
        let mut j = 0;
        let mut res = 0;
        while j < g.len() && i < s.len() {
            if s[i] >= g[j] {
                j += 1;
                i += 1;
                res += 1;
            } else if s[i] < g[j] {
                i += 1;
            }
        }

        res
    }

    //3372
    pub fn max_target_nodes(edges1: Vec<Vec<i32>>, edges2: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
        use std::collections::VecDeque;
        let mut tree1 = vec![vec![]; edges1.len() + 1];
        let mut tree2 = vec![vec![]; edges2.len() + 1];
        let mut asw = vec![0; edges1.len() + 1];
        for edge in edges1 {
            let p = edge[0] as usize;
            let c = edge[1] as usize;
            tree1[p].push(c);
            tree1[c].push(p);
        }

        for edge in edges2 {
            let p = edge[0] as usize;
            let c = edge[1] as usize;
            tree2[p].push(c);
            tree2[c].push(p);
        }

        let mut max_tree2 = 0;
        let mut q = VecDeque::new();
        let mut selected = vec![false; tree2.len()];
        for i in 0..tree2.len() {
            selected.fill(false);
            q.push_back(i);
            selected[i] = true;
            let mut m = 1;
            let mut current_layer = 1;
            if k == 0 {
                max_tree2 = 0;
                break;
            }
            while !q.is_empty() {
                if current_layer >= k {
                    q.clear();
                    break;
                }
                current_layer += 1;
                let mut s = q.split_off(0);

                while let Some(parent) = s.pop_front() {
                    for i in &tree2[parent] {
                        if !selected[*i] {
                            m += 1;
                            q.push_back(*i);
                            selected[*i] = true;
                        }
                    }
                }
            }
            max_tree2 = max_tree2.max(m);
        }

        let mut q = VecDeque::new();
        let mut selected = vec![false; asw.len()];
        for i in 0..asw.len() {
            selected.fill(false);
            let mut max = 0;
            q.push_back(i);
            selected[i] = true;
            let mut layer = 0;

            while !q.is_empty() {
                if layer > k {
                    q.clear();
                    break;
                }
                layer += 1;
                let mut qq = q.split_off(0);
                while let Some(p) = qq.pop_front() {
                    max += 1;

                    for c in &tree1[p] {
                        if !selected[*c] {
                            q.push_back(*c);
                            selected[*c] = true;
                        }
                    }
                }
            }
            asw[i] = max + max_tree2;
        }

        asw
    }

    //456
    pub fn find132pattern(mut nums: Vec<i32>) -> bool {
        use std::collections::BTreeMap;
        let mut left_min = nums[0];
        let mut tree_map = BTreeMap::new();
        for i in 2..nums.len() {
            tree_map.insert(nums[i], *tree_map.get(&nums[i]).unwrap_or(&0) + 1);
        }
        for i in 1..nums.len() - 1 {
            if left_min < nums[i] {
                if tree_map.range(left_min + 1..nums[i]).count() != 0 {
                    return true;
                }
            } else {
                left_min = left_min.min(nums[i]);
            }
            tree_map.insert(nums[i + 1], *tree_map.get(&nums[i + 1]).unwrap() - 1);
            if *tree_map.get(&nums[i + 1]).unwrap() == 0 {
                tree_map.remove(&nums[i + 1]).unwrap();
            }
        }
        false
    }

    //dp 139
    // dp[i] the first i item can be constructed by the word dict
    // dp[i] = true if dict contains s[0..i]
    // check preceding dp
    pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
        let mut dp = vec![false; s.len()];
        let s = s.as_str();
        for i in 0..s.len() {
            if word_dict.contains(&format!("{}", &s[0..=i])) {
                dp[i] = true;
            } else {
                for j in 0..i {
                    if dp[j] && word_dict.contains(&format!("{}", &s[j + 1..=i])) {
                        dp[i] = true
                    }
                }
            }
        }
        dp.pop().unwrap()
    }
    //516
    // dp[i,j] max count from i to j
    // dp[i,j] = max( dp[i + 1,j], dp[i,j - 1] )
    // dp[i,j] = dp[i + 1,j -1] + 2 , if s[i] == s[j]
    pub fn longest_palindrome_subseq(s: String) -> i32 {
        let mut dp = vec![vec![1; s.len()]; s.len()];
        let s = s.as_bytes();
        for i in 0..s.len() - 1 {
            if s[i] == s[i + 1] {
                dp[i][i + 1] = 2
            }
        }
        for i in 2..s.len() {
            for j in 0..s.len() - i {
                // println!("{} {}",j,j+ i);
                if s[j] == s[j + i] {
                    dp[j][j + i] = dp[j + 1][j + i - 1] + 2;
                } else {
                    dp[j][j + i] = dp[j + 1][j + i].max(dp[j][j + i - 1]);
                }
            }
        }
        // println!("{:?}",dp);
        *dp[0].last().unwrap()
    }
    // dp[i,j]
    pub fn min_distance(word1: String, word2: String) -> i32 {
        let word1 = word1.as_bytes();
        let word2 = word2.as_bytes();
        let mut dp = vec![vec![0; word1.len() + 1]; word2.len() + 1];
        for i in 1..dp.len() {
            dp[i][0] = dp[i - 1][0] + 1;
        }
        for i in 1..dp[0].len() {
            dp[0][i] = dp[0][i - 1] + 1;
        }

        for i in 1..dp.len() {
            for j in 1..dp[0].len() {
                if word1[i - 1] == word2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]).max(dp[i - 1][j - 1]) + 1;
                }
            }
        }

        *dp.last().unwrap().last().unwrap()
    }
    //
    pub fn find_number_of_lis(nums: Vec<i32>) -> i32 {
        use std::collections::HashMap;
        let mut map = HashMap::new();

        let mut dp = vec![1; nums.len()];
        map.insert(1, vec![0]);
        let mut max = 1;
        for i in 1..nums.len() {
            for j in 0..i {
                if nums[i] > nums[j] {
                    dp[i] = dp[i].max(dp[j] + 1);
                }
            }
            max = max.max(dp[i]);
            if let Some(v) = map.get_mut(&dp[i]) {
                v.push(i);
            } else {
                map.insert(dp[i], vec![i]);
            }
        }
        let mut res = 0;
        let mut mem: HashMap<usize, i32> = HashMap::new();
        fn dfs_find(
            search: i32,
            map: &HashMap<i32, Vec<usize>>,
            nums: &Vec<i32>,
            // res: &mut i32,
            lax_idx: usize,
            mem: &mut HashMap<usize, i32>,
        ) -> i32 {
            if search == 0 {
                return 1;
            }
            let mut res = 0;
            let nodes = map.get(&search).unwrap();
            for n in nodes {
                if lax_idx == usize::MAX {
                    if let Some(r) = mem.get(&lax_idx) {
                        return *r;
                    } else {
                        res += dfs_find(search - 1, map, nums, *n, mem);
                    }
                } else if *n < lax_idx && nums[*n] < nums[lax_idx] {
                    if let Some(r) = mem.get(&lax_idx) {
                        return *r;
                    } else {
                        res += dfs_find(search - 1, map, nums, *n, mem);
                    }
                }
            }
            mem.insert(lax_idx, res);
            res
        }
        dfs_find(max, &map, &nums, usize::MAX, &mut mem)
    }

    //3373
    pub fn max_target_nodes_v2(edges1: Vec<Vec<i32>>, edges2: Vec<Vec<i32>>) -> Vec<i32> {
        use std::collections::{HashSet, VecDeque};
        let mut tree1 = vec![vec![]; edges1.len() + 1];
        let mut tree2 = vec![vec![]; edges2.len() + 1];
        let mut res = vec![0; edges1.len() + 1];
        // construct tree1 by edges 1
        for edge in edges1 {
            tree1[edge[0] as usize].push(edge[1] as usize);
            tree1[edge[1] as usize].push(edge[0] as usize);
        }

        for edge in edges2 {
            tree2[edge[0] as usize].push(edge[1] as usize);
            tree2[edge[1] as usize].push(edge[0] as usize);
        }
        let mut set_tree2 = HashSet::new();
        for i in 0..tree2.len() {
            set_tree2.insert(i);
        }
        let mut max_target_num_on_tree2 = 0;

        let mut q = VecDeque::new();
        let mut is_selected_tree2 = vec![false; tree2.len()];
        let mut target_nodes_tree2_set = HashSet::new();
        q.push_back(0);
        is_selected_tree2[0] = true;
        let mut layer = 0;
        while !q.is_empty() {
            let parents = q.split_off(0);

            for parent in parents {
                for child in tree2[parent].iter() {
                    if layer % 2 == 0 {
                        target_nodes_tree2_set.insert(*child);
                    }
                    if !is_selected_tree2[*child] {
                        q.push_back(*child);
                        is_selected_tree2[*child] = true;
                    }
                }
            }
            layer += 1;
        }

        let a = set_tree2.difference(&target_nodes_tree2_set).count();

        max_target_num_on_tree2 = a.max(target_nodes_tree2_set.len()) as i32;

        let mut q = VecDeque::new();
        let mut selected_tree1 = vec![false; tree1.len()];
        let mut tree1_set = HashSet::new();
        for i in 0..tree1.len() {
            tree1_set.insert(i);
        }
        q.push_back(0);
        layer = 0;
        let mut target_nodes_set = HashSet::new();
        while !q.is_empty() {
            let mut parents = q.split_off(0);
            for parent in parents {
                if layer % 2 == 0 {
                    target_nodes_set.insert(parent);
                }
                for c in &tree1[parent] {
                    if !selected_tree1[*c] {
                        q.push_back(*c);
                        selected_tree1[*c] = true;
                    }
                }
            }
            layer += 1;
        }

        let b = tree1_set.difference(&target_nodes_set).count() as i32;
        target_nodes_set.iter().for_each(|x| {
            res[*x] = target_nodes_set.len() as i32 + max_target_num_on_tree2;
        });
        tree1_set.difference(&target_nodes_set).for_each(|x| {
            res[*x] = b + max_target_num_on_tree2;
        });
        res
    }
    //457
    pub fn circular_array_loop(nums: Vec<i32>) -> bool {
        fn cal_idx(step: i32, len: usize, current: usize) -> usize {
            ((step % len as i32 + current as i32 + len as i32) % len as i32) as usize
        }
        fn in_the_same_direction(num: i32, dir: bool) -> bool {
            if dir { num > 0 } else { num < 0 }
        }
        let mut fast = 0;
        let mut slow = 0;
        let mut dir;
        for i in 0..nums.len() - 1 {
            fast = i;
            slow = i;
            dir = nums[fast] > 0;
            loop {
                let next1_fast = cal_idx(nums[fast], nums.len(), fast);
                if !in_the_same_direction(nums[next1_fast], dir) {
                    break;
                }
                if fast == next1_fast {
                    break;
                }
                fast = cal_idx(nums[next1_fast], nums.len(), next1_fast);
                if !in_the_same_direction(nums[fast], dir) {
                    break;
                }
                if fast == slow {
                    return true;
                }
                slow = cal_idx(nums[slow], nums.len(), slow);
                if !in_the_same_direction(nums[slow], dir) {
                    break;
                }
            }
        }
        false
    }

    //89 finally!!!!!
    pub fn gray_code(n: i32) -> Vec<i32> {
        if n == 1 {
            vec![0, 1]
        } else {
            let mut res = vec![];
            let next = Self::gray_code(n - 1);
            let x = 1 << n;
            let y = 0 << n;
            for i in 0..next.len() {
                res.push(next[i] & y);
            }
            for i in (0..next.len()).rev() {
                res.push(next[i] & x);
            }
            res
        }
    }
    //241 finally!
    pub fn diff_ways_to_compute(expression: String) -> Vec<i32> {
        use std::collections::HashMap;
        let mut dp = HashMap::new();

        let mut ops = vec![];
        let mut temp = String::new();
        for i in expression.chars() {
            if i.is_numeric() {
                temp.push(i);
            } else {
                ops.push(temp.parse::<i32>().unwrap());
                ops.push(match i {
                    '+' => -1,
                    '-' => -2,
                    '*' => -3,
                    _ => {
                        unreachable!()
                    }
                });
                temp.clear();
            }
        }

        fn cal(l: i32, o: i32, r: i32) -> i32 {
            match o {
                -1 => l + o,
                -2 => l - o,
                -3 => l * o,
                _ => unreachable!(),
            }
        }
        fn dfs(
            ops: &[i32],
            start: usize,
            end: usize,
            dp: &mut HashMap<(usize, usize), Vec<i32>>,
        ) -> Vec<i32> {
            println!("{:?}", (start, end));
            if let Some(cache) = dp.get(&(start, end)) {
                return cache.clone();
            }

            if end - start == 1 {
                return vec![ops[0]];
            }
            if end - start == 3 {
                return vec![cal(ops[start], ops[start + 1], ops[start + 2])];
            }
            let mut res = vec![];
            for i in (start + 1..end).step_by(2) {
                let left = dfs(&ops, start, i, dp);
                let right = dfs(&ops, i + 1, end, dp);
                for l in left {
                    for r in &right {
                        res.push(cal(l, ops[i], *r));
                    }
                }
            }
            dp.insert((start, end), res.clone());
            res
        }
        dfs(&ops, 0, ops.len(), &mut dp)
    }
    //2359
    pub fn closest_meeting_node(edges: Vec<i32>, node1: i32, node2: i32) -> i32 {
        use std::collections::VecDeque;
        let mut nodes = vec![vec![]; edges.len()];
        let mut res = usize::MAX;
        let mut dest_from_node1 = vec![nodes.len(); edges.len()];
        let mut dest_from_node2 = vec![nodes.len(); edges.len()];

        for (idx, value) in edges.into_iter().enumerate() {
            if value >= 0 {
                nodes[idx].push(value as usize);
            }
        }

        dest_from_node1[node1 as usize] = 0;
        dest_from_node2[node2 as usize] = 0;

        fn dest_(dest: &mut Vec<usize>, node: usize, nodes: &Vec<Vec<usize>>) {
            let mut q = VecDeque::new();
            let mut selected = vec![false; nodes.len()];

            for i in &nodes[node] {
                q.push_back(*i);
                selected[*i] = true;
            }

            let mut path_len = 1;
            while !q.is_empty() {
                let nexts = q.split_off(0);
                for n in nexts {
                    dest[n] = dest[n].min(path_len);
                    for nn in &nodes[n] {
                        if !selected[*nn] {
                            q.push_back(*nn);
                            selected[*nn] = true;
                        }
                    }
                }
                path_len += 1;
            }
        }

        dest_(&mut dest_from_node1, node1 as usize, &nodes);
        dest_(&mut dest_from_node2, node2 as usize, &nodes);
        println!("{:?}", dest_from_node1);
        println!("{:?}", dest_from_node2);
        let mut min = usize::MAX;
        for i in 0..dest_from_node1.len() {
            if dest_from_node1[i] != nodes.len() && dest_from_node2[i] != nodes.len() {
                if min > dest_from_node1[i].max(dest_from_node2[i]) {
                    min = dest_from_node1[i].max(dest_from_node2[i]);
                    res = i;
                }
            }
        }
        if res == usize::MAX { -1 } else { res as i32 }
    }

    //459
    pub fn repeated_substring_pattern(s: String) -> bool {
        let s = s.as_str();
        if s.len() == 1 {
            return false;
        }
        for len in 1..s.len() / 2 {
            let m = &s[..len];
            for i in (0..s.len()).step_by(len) {
                if &s[i..(i + len)] != m {
                    break;
                }
                if &s[i..(i + len)] == m && i + len == s.len() {
                    return true;
                }
            }
        }
        false
    }
    //461
    pub fn hamming_distance(x: i32, y: i32) -> i32 {
        let mut res = 0;
        for i in 0..=31 {
            if x & (1 << i) != y & (1 << i) {
                res += 1;
            }
        }
        res
    }
    //462
    pub fn min_moves2(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        nums.sort();
        let mid = nums[nums.len() / 2];
        let mut res = 0;
        for i in nums {
            res += (i - mid).abs();
        }
        res
    }

    //463
    pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
        fn search(grid: &Vec<Vec<i32>>, mark: &mut Vec<Vec<bool>>, row: usize, col: usize) -> i32 {
            if mark[row][col] {
                return 0;
            }
            if grid[row][col] == 0 {
                return 1;
            }
            let mut res = 0;
            mark[row][col] = true;
            if row == 0 {
                //top
                res += 1;
            } else {
                res += search(grid, mark, row - 1, col);
            }
            if col == 0 {
                //left
                res += 1;
            } else {
                res += search(grid, mark, row, col - 1);
            }
            if row == grid.len() - 1 {
                //bottom
                res += 1;
            } else {
                res += search(grid, mark, row + 1, col);
            }
            if col == grid[0].len() - 1 {
                res += 1;
            } else {
                res += search(grid, mark, row, col + 1);
            }

            res
        }
        let mut mark: Vec<Vec<bool>> = vec![vec![false; grid[0].len()]; grid.len()];
        for i in 0..grid.len() {
            for j in 0..grid[0].len() {
                if grid[i][j] == 1 {
                    return search(&grid, &mut mark, i, j);
                }
            }
        }
        unreachable!()
    }

    //464 the fuck!
    pub fn can_i_win(max_choosable_integer: i32, desired_total: i32) -> bool {
        let mut numbers = vec![false; max_choosable_integer as usize];

        fn dfs(selected: &mut Vec<bool>, current: i32, is_my_own: bool, target: i32) -> bool {
            if current >= target {
                return true;
            }
            for i in 0..selected.len() {
                if !selected[i] {
                    selected[i] = true;
                    let r = dfs(selected, current + i as i32 + 1, !is_my_own, target);
                    selected[i] = false;
                    if r {
                        return r;
                    }
                }
            }
            false
        }

        for i in 0..max_choosable_integer as usize {
            numbers[i] = true;
            if dfs(&mut numbers, i as i32 + 1, true, desired_total) {
                return true;
            }
            numbers[i] = false;
        }

        false
    }

    //909
    pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
        use std::collections::VecDeque;
        let mut b = vec![];
        let mut board = board;
        board.reverse();
        for row in 0..board.len() {
            if row % 2 == 1 {
                board[row].reverse();
            }
            b.append(&mut board[row]);
        }
        let mut selected = vec![false; b.len()];
        let mut q = VecDeque::new();
        q.push_back(0);
        selected[0] = true;
        let mut times = 0;
        while !q.is_empty() {
            let pq = q.split_off(0);
            times += 1;
            for next in pq {
                for i in 1..=6 {
                    if next + i >= b.len() - 1 {
                        return times;
                    }
                    if next + i < selected.len() && !selected[next + i] {
                        if b[next + i] == -1 {
                            q.push_back(next + i);
                        } else {
                            q.push_back(b[next + i] as usize - 1);
                        }
                        selected[next + i] = true;
                    }
                }
            }
        }
        -1
    }
    //467 dp[i], a-z; 以i 结尾的个数 最长的！
    pub fn find_substring_in_wrapround_string(s: String) -> i32 {
        let mut dp = vec![0; 26];
        let s = s.as_bytes();
        let mut k = 1;
        for i in 0..s.len() {
            if i > 0 && s[i] - b'a' == (s[i - 1] - b'a' + 1) % 26 {
                k += 1;
            } else {
                k = 1;
            }
            dp[(s[i] - b'a') as usize] = dp[(s[i] - b'a') as usize].max(k);
        }
        dp.into_iter().sum()
    }

    pub fn check_equal_partitions(mut nums: Vec<i32>, target: i64) -> bool {
        use std::collections::HashSet;
        fn dfs_search(
            size: usize,
            nums: &Vec<i32>,
            target: i64,
            current: &mut HashSet<i32>,
            start: usize,
            current_value: i64,
        ) -> bool {
            // println!("{:?}",current );
            if current.len() == size {
                // println!("{:?}",current );
                if current_value == target {
                    let mut c = 1;
                    for i in nums.iter().filter(|x| !current.contains(x)) {
                        c *= *i as i64;
                    }
                    if c == target {
                        return true;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            for i in start..nums.len() {
                let x = current_value * nums[i] as i64;
                if x > target {
                    break;
                }
                current.insert(nums[i]);
                if dfs_search(size, nums, target, current, start + 1, x) {
                    return true;
                }
                current.remove(&nums[i]);
            }
            false
        }
        let mut nums = nums;
        nums.sort();
        let mut current = HashSet::new();
        for i in 1..=nums.len() / 2 {
            if dfs_search(i, &nums, target, &mut current, 0, 1) {
                return true;
            }
        }
        false
    }

    // c31: n: 3,x: 1;
    pub fn conbination(n: Vec<i32>, x: usize) {
        let mut selected = vec![false; n.len()];

        fn find(
            nums: &Vec<i32>,
            current: &mut Vec<i32>,
            seleted: &mut Vec<bool>,
            x: usize,
            start: usize,
        ) {
            if x == 0 {
                println!("{:?}", current);
                return;
            }
            for i in start..nums.len() {
                current.push(nums[i]);
                find(nums, current, seleted, x - 1, i + 1);
                current.pop();
            }
        }

        let mut s = vec![];
        find(&n, &mut s, &mut selected, x, 0);
    }

    //q2
    pub fn min_abs_diff(grid: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
        let k = k as usize;
        let mut res = vec![vec![0; grid[0].len() - k + 1]; grid.len() - k + 1];
        for i in 0..=grid.len() - k {
            for j in 0..=grid[0].len() - k {
                let mut s = vec![];
                for row in i + 0..i + k {
                    for col in j + 0..j + k {
                        s.push(grid[row][col]);
                    }
                }
                s.sort();
                s.dedup();
                let mut min = i32::MAX;
                for i in 1..s.len() {
                    if (s[i] - s[i - 1]).abs() < min {
                        min = (s[i] - s[i - 1]).abs();
                    }
                }
                res[i][j] = min;
            }
        }
        res
    }

    //468
    pub fn valid_ip_address(query_ip: String) -> String {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        for i in 0..9 {
            set.insert(b'0' + i);
        }
        for i in 0..6 {
            set.insert(b'a' + i);
        }
        for i in 0..6 {
            set.insert(b'A' + i);
        }
        if query_ip.contains(".") {
            let mut q = query_ip.split(".");
            if q.clone().count() != 4 {
                return "Neither".to_string();
            }

            if q.all(|x| {
                if x.len() != 1 && x.starts_with("0") || x.len() == 0 {
                    return false;
                } else {
                    if let Ok(p) = x.parse::<i32>() {
                        if p < 0 || p > 255 {
                            return false;
                        } else {
                            return true;
                        }
                    } else {
                        return false;
                    }
                }
            }) {
                return "IPv4".to_string();
            } else {
                return "Neither".to_string();
            }
        } else {
            if query_ip.split(":").count() != 8 {
                return "Neither".to_string();
            }
            if query_ip.split(":").all(|x| {
                if x.len() > 4 || x.len() == 0 {
                    return false;
                }
                if x.chars().any(|xx| !set.contains(&(xx as u8))) {
                    return false;
                }
                true
            }) {
                return "IPv6".to_string();
            } else {
                "Neither".to_string()
            }
        }
    }
    //473
    pub fn makesquare(mut matchsticks: Vec<i32>) -> bool {
        let mut matchsticks = matchsticks;

        let sum = matchsticks.iter().sum::<i32>();

        if sum % 4 != 0 {
            return false;
        }

        matchsticks.sort_by(|a, b| b.cmp(a));
        let mut edges = vec![0; 4];

        fn dfs(edges: &mut Vec<i32>, matchsticks: &Vec<i32>, index: usize, n: i32) -> bool {
            if index == matchsticks.len() {
                return true;
            }
            for i in 0..edges.len() {
                edges[i] += matchsticks[index];
                if edges[i] <= n && dfs(edges, matchsticks, index + 1, n) {
                    return true;
                }
                edges[i] -= matchsticks[index];
            }
            false
        }
        dfs(&mut edges, &matchsticks, 0, sum / 4)
    }

    //e135
    pub fn candy(mut ratings: Vec<i32>) -> i32 {
        let mut i = 1;
        let mut r = vec![i32::MAX];
        r.append(&mut ratings);
        let ratings = r;
        let mut res = vec![0; ratings.len()];

        loop {
            if ratings[i] > ratings[i - 1] && ratings[i] <= ratings[i + 1] {
                res[i] = res[i - 1] + 1;
            } else {
                let mut times = 1;
                let mut p_i = i;
                while p_i < ratings.len() - 1 && ratings[p_i] > ratings[p_i + 1] {
                    times += 1;
                    p_i += 1;
                }
                // println!("{} {} ",res[i - 1],times);
                if times < res[i - 1] + 1 && ratings[i] != ratings[i - 1] {
                    res[i] = res[i - 1] + 1;
                    i += 1;
                    times -= 1;
                    while i <= p_i {
                        res[i] = times;
                        i += 1;
                        times -= 1;
                    }
                } else {
                    while i <= p_i {
                        res[i] = times;
                        i += 1;
                        times -= 1;
                    }
                }

                i -= 1;
            }
            i += 1;
            if i >= res.len() - 1 {
                break;
            }
        }
        if ratings[ratings.len() - 1] > ratings[ratings.len() - 2] {
            res[ratings.len() - 1] = res[res.len() - 2] + 1;
        } else {
            res[ratings.len() - 1] = 1;
        }
        let res = res.iter().sum::<i32>();
        res
    }
    //337 dp
    pub fn rob(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(root: Option<Rc<RefCell<TreeNode>>>) -> (i32, i32) {
            if let Some(r) = root {
                let root = r.borrow();
                if root.left.is_none() && root.right.is_none() {
                    return (root.val, 0);
                }
                let left = dfs(root.left.clone());
                let right = dfs(root.right.clone());
                let v = root.val;
                (
                    (left.0 + right.0).max(left.1 + right.1 + v),
                    (left.0 + right.0).max(left.1 + right.1),
                )
            } else {
                (0, 0)
            }
        }
        let res = dfs(root);
        res.0.max(res.1)
    }

    //dp
    pub fn num_squares(n: i32) -> i32 {
        let mut factors = vec![];
        for i in 1.. {
            if i * i <= n {
                factors.push((i * i) as usize);
            } else {
                break;
            }
        }
        let mut dp = vec![vec![0; factors.len()]; n as usize + 1];
        for i in 0..dp.len() {
            dp[i][0] = 0;
        }
        for i in 1..dp[0].len() {
            dp[0][i] = dp[0][i - 1] + 1;
        }
        for i in 1..dp.len() {
            for j in 1..dp[0].len() {
                if j > factors[i] {
                    dp[i][j] = dp[i - 1][j]
                } else {
                    dp[i][j] = (dp[i][j - factors[i]] + 1).min(dp[i - 1][j]);
                }
            }
        }

        *dp.last().unwrap().last().unwrap()
    }

    //518
    pub fn change(amount: i32, coins: Vec<i32>) -> i32 {
        let mut dp = vec![vec![amount as usize + 1]; coins.len()];
        for i in 0..dp.len() {
            dp[i][0] = 1;
        }
        for i in 1..dp[0].len() {
            if i >= coins[0] as usize && i % coins[0] as usize == 0 {
                dp[0][i] = 1;
            }
        }
        for i in 1..dp.len() {
            for j in 1..dp[0].len() {
                if j < coins[i] as usize {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i] as usize];
                }
            }
        }
        *dp.last().unwrap().last().unwrap() as i32
    }
    //377 review
    pub fn combination_sum4(nums: Vec<i32>, target: i32) -> i32 {
        let mut res = 0;
        let mut nums = nums;
        nums.sort();
        fn dfs(target: i32, nums: &Vec<i32>, current: i32, res: &mut i32) {
            for i in 0..nums.len() {
                let mut next = current + nums[i];
                if next == target {
                    *res += 1;
                    break;
                }
                if next > target {
                    break;
                }
                dfs(target, nums, next, res);
            }
        }
        dfs(target, &nums, 0, &mut res);
        res
    }

    //474
    pub fn find_max_form(strs: Vec<String>, m: i32, n: i32) -> i32 {
        let x: Vec<(usize, usize)> = strs
            .iter()
            .map(|x| {
                let one = x.chars().filter(|y| *y == '1').count();
                (one, x.len() - one)
            })
            .collect();
        let mut dp = vec![vec![vec![0; m as usize + 1]; n as usize + 1]; strs.len()];
        // for i in 0..strs.len() {
        //     dp[i][0][0] = 0;
        // }
        for i in 0..m as usize + 1 {
            for j in 0..n as usize + 1 {
                dp[0][j][i] = if x[0].0 > j || x[0].1 > i { 0 } else { 1 };
            }
        }
        for i in 1..strs.len() {
            for j in 0..n as usize + 1 {
                for k in 0..m as usize + 1 {
                    if x[i].0 > j || x[i].1 > k {
                        if i == 1 {}
                        dp[i][j][k] = dp[i - 1][j][k];
                    } else {
                        if i == 1 {}
                        dp[i][j][k] = dp[i - 1][j][k].max(dp[i - 1][j - x[i].0][k - x[i].1] + 1);
                    }
                }
            }
        }
        // for i in &dp {
        // }
        *dp.last().unwrap().last().unwrap().last().unwrap()
    }

    pub fn max_candies(
        status: Vec<i32>,
        candies: Vec<i32>,
        keys: Vec<Vec<i32>>,
        contained_boxes: Vec<Vec<i32>>,
        initial_boxes: Vec<i32>,
    ) -> i32 {
        use std::collections::{HashMap, HashSet, VecDeque};
        let mut locked_boxes = HashSet::new();
        let mut owned_keys = VecDeque::new();
        let mut res = 0;
        let mut opened_boxes = VecDeque::new();
        for i in 0..1 {}
        for b in initial_boxes {
            if status[b as usize] == 1 {
                //open
                opened_boxes.push_back(b as usize);
            } else {
                locked_boxes.insert(b as usize);
            }
        }
        while !opened_boxes.is_empty() || owned_keys.iter().any(|x| locked_boxes.contains(x)) {
            let mut p_o_b = opened_boxes.split_off(0);
            for b in p_o_b {
                res += candies[b];
                for k in &keys[b] {
                    owned_keys.push_back(*k as usize);
                }
                for new_b in &contained_boxes[b] {
                    if status[*new_b as usize] == 1 {
                        opened_boxes.push_back(*new_b as usize);
                    } else {
                        locked_boxes.insert(*new_b as usize);
                    }
                }
            }
            let mut o_k = owned_keys.split_off(0);
            for k in o_k {
                if locked_boxes.remove(&k) {
                    //get opened!
                    res += candies[k];
                    for k in &keys[k] {
                        owned_keys.push_back(*k as usize);
                    }
                    for new_b in &contained_boxes[k] {
                        if status[*new_b as usize] == 1 {
                            opened_boxes.push_back(*new_b as usize);
                        } else {
                            locked_boxes.insert(*new_b as usize);
                        }
                    }
                } else {
                    owned_keys.push_back(k);
                }
            }
        }
        res
    }
}
