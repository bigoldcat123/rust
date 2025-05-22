use std::{cell::RefCell, process::id, rc::Rc};

use rand::rand_core::le;

use crate::{ListNode, TreeNode};

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
    //433
    pub fn min_mutation(start_gene: String, end_gene: String, bank: Vec<String>) -> i32 {
        use std::collections::{HashMap, HashSet};
        if bank.is_empty() {
            return -1;
        }
        if !bank.iter().any(|x| x == &end_gene) {
            return -1;
        }

        fn can_convert(from: &[u8], to: &[u8]) -> bool {
            let mut diff = 0;
            for i in 0..8 {
                if from[i] != to[i] {
                    diff += 1
                }
            }
            diff == 1
        }
        fn dfs_search<'a>(
            bank: &'a Vec<String>,
            selected: &mut HashSet<&'a String>,
            current_gent: &String,
            start_gene: &String,
            res: &mut i32,
        ) {
            if can_convert(current_gent.as_bytes(), start_gene.as_bytes()) {
                println!("{:?}", selected);
                let len = selected.len() as i32;
                *res = (*res).min(len + 1);
                return;
            }
            for b in bank {
                if dbg!(can_convert(current_gent.as_bytes(), b.as_bytes())) && !selected.contains(b)
                {
                    selected.insert(&b);
                    dfs_search(bank, selected, b, start_gene, res);
                    selected.remove(&b);
                }
            }
        }
        let mut selected = HashSet::new();
        let mut res = i32::MAX;
        dfs_search(&bank, &mut selected, &end_gene, &start_gene, &mut res);
        if res == i32::MAX { -1 } else { res }
    }
    //434
    pub fn count_segments(s: String) -> i32 {
        s.split_whitespace().count() as i32
    }
    //435
    pub fn erase_overlap_intervals(intervals: Vec<Vec<i32>>) -> i32 {
        // let mut intervals = intervals;
        // intervals.sort_by(|a, b| a[0].cmp(&b[0]));
        // let mut dp = vec![1; intervals.len()];
        // for i in 1..intervals.len() {
        //     for j in 0..i {
        //         if intervals[j][1] <= intervals[i][0] {
        //             dp[i] = dp[i].max(dp[j] + 1);
        //         }
        //     }
        // }
        // (intervals.len() - *dp.last().unwrap()) as i32

        let mut intervals = intervals;
        intervals.sort_by(|a, b| a[1].cmp(&b[1]));
        let mut right = intervals[0][1];
        let mut res = 1;
        for i in 1..intervals.len() {
            if intervals[i][0] >= right {
                right = intervals[i][1];
                res += 1;
            }
        }
        intervals.len() as i32 - res
    }
    //436
    pub fn find_right_interval(intervals: Vec<Vec<i32>>) -> Vec<i32> {
        use std::collections::HashMap;
        let mut sorted_intervals = intervals.clone();
        sorted_intervals.sort_by(|a, b| a[0].cmp(b.first().unwrap()));
        let mut map = HashMap::new();
        for (idx, value) in intervals.iter().enumerate() {
            map.insert(value, idx);
        }

        let mut dp = HashMap::new();
        for i in 0..sorted_intervals.len() {
            let right = sorted_intervals[i][1];
            for j in i..sorted_intervals.len() {
                if sorted_intervals[j][0] >= right {
                    dp.insert(&sorted_intervals[i], &sorted_intervals[j]);
                    break;
                }
            }
        }
        let mut res = vec![0; sorted_intervals.len()];
        for i in 0..sorted_intervals.len() {
            let origin_iterval = &intervals[i];
            let target_interval = dp.remove(origin_iterval);
            match target_interval {
                None => {
                    res[i] = -1;
                }
                Some(target) => {
                    res[i] = *map.get(target).unwrap() as i32;
                }
            }
        }
        res
    }

    //437
    pub fn path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> i32 {
        match root {
            Some(root) => {
                let mut res = 0;
                // if root.borrow().val == target_sum {
                //     res += 1;
                // }
                res += Self::path_sum(root.borrow().left.clone(), target_sum);
                res += Self::path_sum(root.borrow().right.clone(), target_sum);
                res += Self::path_sum_dfs(Some(root), target_sum as usize, 0);

                res
            }
            None => 0,
        }
    }
    pub fn path_sum_dfs(
        root: Option<Rc<RefCell<TreeNode>>>,
        target_sum: usize,
        current: usize,
    ) -> i32 {
        match root {
            None => 0,
            Some(root) => {
                let mut plus = 0;
                let root = root.borrow();
                let current = root.val as usize + current;

                if current == target_sum {
                    plus += 1;
                }
                let left_sum = Self::path_sum_dfs(root.left.clone(), target_sum, current);
                let right_sum = Self::path_sum_dfs(root.right.clone(), target_sum, current);
                left_sum + right_sum + plus
            }
        }
    }
    //438
    pub fn find_anagrams(s: String, p: String) -> Vec<i32> {
        let mut res = vec![];
        if s.len() < p.len() {
            return res;
        }
        let s = s.as_bytes();
        let p = p.as_bytes();
        let mut s_number = [0; 26];
        let mut p_number = [0; 26];
        for i in 0..p.len() {
            let s_char = s[i];
            let p_char = p[i];
            s_number[(s_char - b'a') as usize] += 1;
            p_number[(p_char - b'a') as usize] += 1;
        }
        if s_number == p_number {
            res.push(0);
        }
        for i in 1..s.len() - p.len() + 1 {
            s_number[(s[i - 1] - b'a') as usize] -= 1;
            s_number[(s[i + p.len() - 1] - b'a') as usize] += 1;
            if s_number == p_number {
                res.push(i as i32);
            }
        }

        res
    }
    //442
    pub fn find_duplicates(nums: Vec<i32>) -> Vec<i32> {
        let len = nums.len();
        let mut nums = nums;
        let mut i = 0;
        loop {
            if i == len {
                break;
            }
            while nums[i] != nums[nums[i] as usize - 1] {
                let p = nums[i];
                let next_idx = nums[i] as usize - 1;
                nums[i] = nums[next_idx];
                nums[next_idx] = p;
            }
            i += 1;
        }
        let mut res = vec![];
        for i in 0..len {
            if nums[i] == i as i32 + 1 {
                res.push(nums[i]);
            }
        }
        res
    }

    // 443
    pub fn compress(chars: &mut Vec<char>) -> i32 {
        let mut left = 0;
        let mut right = 0;
        let mut current_len = 0;
        let mut res_char = vec![];
        while right < chars.len() {
            if chars[left] == chars[right] {
                current_len += 1;
            } else {
                res_char.push(chars[left]);
                if current_len > 1 {
                    for e in current_len.to_string().chars() {
                        res_char.push(e);
                    }
                }

                left = right;
                current_len = 0;
                continue;
            }
            right += 1;
        }
        *chars = res_char;
        chars.len() as i32
    }
    //445
    pub fn add_two_numbers(
        l1: Option<Box<ListNode>>,
        l2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        fn list_len(l: Option<&Box<ListNode>>, len: &mut i32) {
            match l {
                Some(l) => {
                    *len += 1;
                    list_len(l.next.as_ref(), len);
                }
                _ => {}
            }
        }
        let mut len1 = 0;
        let mut len2 = 0;
        let mut res = Some(Box::new(ListNode::new(0)));
        list_len(l1.as_ref(), &mut len1);
        list_len(l2.as_ref(), &mut len2);
        if len1 != len2 {
            let l_need_append;
            let another;
            if len1 > len2 {
                l_need_append = l2;
                another = l1;
            } else {
                l_need_append = l1;
                another = l2;
            }
            let dif = (len1 - len2).abs();
            let mut new_h = ListNode::new(0);
            let mut x = &mut new_h;
            for _ in 1..dif {
                // let a = &mut new_node;
                x.next = Some(Box::new(ListNode::new(0)));
                x = x.next.as_mut().unwrap();
            }
            x.next = l_need_append;

            let carry = dfs_build(
                another.as_ref(),
                Some(Box::new(new_h)).as_ref(),
                res.as_mut(),
            );
            if carry != 0 {
                res.as_mut().unwrap().val = carry;
                return res;
            }
            return res.unwrap().next;
        } else {
            let mut res = Some(Box::new(ListNode::new(0)));
            let carry = dfs_build(l2.as_ref(), l1.as_ref(), res.as_mut());
            if carry != 0 {
                res.as_mut().unwrap().val = carry;
                return res;
            }
            return res.unwrap().next;
        }
        fn dfs_build(
            l1: Option<&Box<ListNode>>,
            l2: Option<&Box<ListNode>>,
            res: Option<&mut Box<ListNode>>,
        ) -> i32 {
            match (l1, l2) {
                (Some(l1), Some(l2)) => {
                    let mut node = Some(Box::new(ListNode::new(0)));
                    let carry = dfs_build(l1.next.as_ref(), l2.next.as_ref(), node.as_mut());
                    let r = l1.val + l2.val + carry;
                    node.as_mut().unwrap().val = r % 10;
                    res.unwrap().next = node;
                    r / 10
                }
                (None, None) => 0,
                _ => {
                    unreachable!()
                }
            }
        }
    }
}
