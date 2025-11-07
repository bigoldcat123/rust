use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    io::Read,
    num,
    os::macos::raw::stat,
    ptr,
};

use crate::ListNode;
pub fn max_power(stations: Vec<i32>, r: i32, k: i32) -> i64 {
    let mut diff = vec![0; stations.len() + 1];
    for (i, &s) in stations.iter().enumerate() {
        let start = (i as i32 - r).max(0) as usize;
        let end = (i + r as usize + 1).min(stations.len());
        diff[start] += s as i64;
        diff[end] -= s as i64;
    }
    let mut min = stations.iter().min().copied().unwrap() as i64;
    let mut max = (stations.iter().max().copied().unwrap() as i64) + k as i64;
    while min <= max {
        let mid = (max - min) / 2 + min;
        if check(diff.as_ref(),mid,r as usize,k as i64) {
            min = mid + 1;
        }else {
            max = mid - 1;
        }
    }
    max
}
fn  check(diff:&[i64],target:i64,radius:usize,mut k:i64) -> bool {
    let mut d = vec![0;diff.len()];
    let mut current = 0;
    for i in 0..diff.len() - 1 {
        current += diff[i] + d[i];
        let need = target - current;
        if need > 0 {
           if  k >= need  {
               current = target;
               d[i + 2 * radius + 1] -= need;
               k -= need;
           }else {
               return false
           }
        }
    }
    k >= 0
}
pub fn find_x_sum2(nums: Vec<i32>, k: i32, x: i32) -> Vec<i64> {
    let x = x as i64;
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    let mut num_occurance_map: HashMap<i64, i64> = HashMap::new();
    let mut sum = 0;
    let mut max_accurance_nums_map: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let mut unsued_accurance_nums_map: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let mut used = 0;
    for i in 0..k as usize {
        *num_occurance_map.entry(nums[i] as i64).or_default() += 1;
    }
    for (&k, &v) in num_occurance_map.iter() {
        if used < x {
            max_accurance_nums_map.entry(v).or_default().insert(k);
            sum += k * v;
            used += 1;
        } else {
            if let Some(mut e) = max_accurance_nums_map.first_entry() {
                if *e.key() < v {
                    sum -= *e.key() * e.get().first().copied().unwrap();
                    unsued_accurance_nums_map
                        .entry(*e.key())
                        .or_default()
                        .insert(e.get_mut().pop_first().unwrap());
                    if e.get().len() == 0 {
                        max_accurance_nums_map.pop_first();
                    }
                    max_accurance_nums_map.entry(v).or_default().insert(k);
                    sum += v * k;
                } else if *e.key() == v {
                    if *e.get().first().unwrap() < k {
                        sum -= *e.key() * e.get().first().copied().unwrap();
                        unsued_accurance_nums_map
                            .entry(*e.key())
                            .or_default()
                            .insert(e.get_mut().pop_first().unwrap());
                        if e.get().len() == 0 {
                            max_accurance_nums_map.pop_first();
                        }
                        sum += v * k;
                        max_accurance_nums_map.entry(v).or_default().insert(k);
                    } else {
                        unsued_accurance_nums_map.entry(v).or_default().insert(k);
                    }
                } else {
                    unsued_accurance_nums_map.entry(v).or_default().insert(k);
                }
            }
        }
    }

    let mut res = vec![sum];

    for i in k as usize..nums.len() {
        let delete_n = nums[i - k as usize] as i64;
        let new_n = nums[i] as i64;
        let pre_occ_del = *num_occurance_map.entry(delete_n).or_default();
        let pre_occ_new = *num_occurance_map.entry(new_n).or_default();
        *num_occurance_map.entry(delete_n).or_default() -= 1;
        *num_occurance_map.entry(new_n).or_default() += 1;
        if delete_n == new_n {
            res.push(sum);
            continue;
        }
        let mut max_minus = false;
        let mut unsed_add = false;
        if let Some(max_occ) = max_accurance_nums_map.get_mut(&pre_occ_del) {
            if max_occ.remove(&delete_n) {
                max_minus = true;
                sum -= delete_n;
                if max_occ.len() == 0 {
                    max_accurance_nums_map.remove(&pre_occ_del);
                }
                if pre_occ_del - 1 > 0 {
                    max_accurance_nums_map
                        .entry(pre_occ_del - 1)
                        .or_default()
                        .insert(delete_n);
                } else {
                    used -= 1;
                }
            }
        }
        if let Some(max_occ) = unsued_accurance_nums_map.get_mut(&pre_occ_del) {
            if max_occ.remove(&delete_n) {
                sum -= delete_n;
                if max_occ.len() == 0 {
                    unsued_accurance_nums_map.remove(&pre_occ_del);
                }
                if pre_occ_del - 1 > 0 {
                    unsued_accurance_nums_map
                        .entry(pre_occ_del - 1)
                        .or_default()
                        .insert(delete_n);
                }
            }
        }
        let mut added = false;

        if let Some(mac_occ) = max_accurance_nums_map.get_mut(&pre_occ_new) {
            if mac_occ.remove(&new_n) {
                if mac_occ.len() == 0 {
                    max_accurance_nums_map.remove(&pre_occ_new);
                }
                added = true;
                max_accurance_nums_map
                    .entry(pre_occ_new + 1)
                    .or_default()
                    .insert(new_n);
                sum += new_n;
            }
        }

        if let Some(mac_occ) = unsued_accurance_nums_map.get_mut(&pre_occ_new) {
            if mac_occ.remove(&new_n) {
                if mac_occ.len() == 0 {
                    unsued_accurance_nums_map.remove(&pre_occ_new);
                }
                unsed_add = true;
                added = true;
                unsued_accurance_nums_map
                    .entry(pre_occ_new + 1)
                    .or_default()
                    .insert(new_n);
                sum += new_n;
            }
        }
        if !added {
            unsued_accurance_nums_map
                .entry(pre_occ_new + 1)
                .or_default()
                .insert(new_n);
        }
        println!("{:?}", unsued_accurance_nums_map);

        if used < x {
            let mut last = unsued_accurance_nums_map.last_entry().unwrap();
            let &occ = last.key();
            let n = last.get_mut().pop_last().unwrap();
            max_accurance_nums_map.entry(occ).or_default().insert(n);
            sum += n * occ;
            if last.get().len() == 0 {
                unsued_accurance_nums_map.pop_last();
            }
        } else {
            if max_minus || unsed_add {
                if let Some(mut e_max_min) = max_accurance_nums_map.first_entry() {
                    if let Some(mut e_min_max) = unsued_accurance_nums_map.last_entry() {
                        if e_max_min.key() < e_min_max.key() {
                            if e_max_min.get().len() == 1 {
                                let mut r = max_accurance_nums_map.pop_first().unwrap();
                                sum -= r.0 * r.1.last().copied().unwrap();
                                let new = e_min_max.get_mut().pop_last().unwrap();
                                sum += new * *e_min_max.key();
                                max_accurance_nums_map
                                    .entry(*e_min_max.key())
                                    .or_default()
                                    .insert(new);
                                if e_min_max.get().len() == 0 {
                                    unsued_accurance_nums_map.pop_last();
                                }
                                unsued_accurance_nums_map
                                    .entry(r.0)
                                    .or_default()
                                    .insert(r.1.pop_first().unwrap());
                            } else {
                                let k = *e_max_min.key();
                                let min_one = e_max_min.get_mut().pop_first().unwrap();
                                sum -= min_one * k;
                                let new = e_min_max.get_mut().pop_last().unwrap();
                                sum += new * *e_min_max.key();
                                max_accurance_nums_map
                                    .entry(*e_min_max.key())
                                    .or_default()
                                    .insert(new);
                                if e_min_max.get().len() == 0 {
                                    unsued_accurance_nums_map.pop_last();
                                }
                                unsued_accurance_nums_map
                                    .entry(k)
                                    .or_default()
                                    .insert(min_one);
                            }
                        } else if e_max_min.key() == e_min_max.key() {
                            if e_max_min.get().first().unwrap() < e_min_max.get().last().unwrap() {
                                let a = e_max_min.get_mut().pop_first().unwrap();
                                let b = e_min_max.get_mut().pop_last().unwrap();
                                sum -= a * *e_max_min.key();
                                sum += b * *e_max_min.key();
                                e_max_min.get_mut().insert(b);
                                e_min_max.get_mut().insert(a);
                                if e_min_max.get().len() == 0 {
                                    unsued_accurance_nums_map.pop_last();
                                }
                                if e_max_min.get().len() == 0 {
                                    max_accurance_nums_map.pop_first();
                                }
                            }
                        }
                    }
                }
            }
        }

        res.push(sum);
    }

    res
}
pub fn min_cost(colors: String, needed_time: Vec<i32>) -> i32 {
    let mut l = 0;
    let mut r = 1;
    let colors = colors.as_bytes();
    let mut res = 0;
    while r < needed_time.len() {
        let mut t = needed_time[l];
        let mut max_t = t;
        while r < needed_time.len() && colors[r] == colors[l] {
            t += needed_time[r];
            max_t = max_t.max(needed_time[r]);
            r += 1;
        }
        res += t - max_t;
        l = r;
        r + 1;
    }
    res
}

pub fn lex_palindromic_permutation(s: String, target: String) -> String {
    let mut map = vec![0; 26];
    for c in s.chars() {
        let idx = c as u8 as usize - 97;
        map[idx] += 1;
    }
    let odd_num = 0;
    if map.iter().filter(|&x| x % 2 == 1).count() > 1 {
        return "".into();
    }
    let mut current = vec![0; s.len()];
    let mut end_idx = s.len() / 2; // excuusive
    let target = target.as_bytes();
    let mut mid_bigger = false;
    for (ii, i) in map.iter().enumerate() {
        if i % 2 == 1 {
            let idx = (s.len() - i) / 2;
            end_idx = idx;
            for j in idx..i + idx {
                current[j] = (ii + 97) as u8;
                if target[j] < current[j] {
                    mid_bigger = true;
                }
            }
            map[ii] = 0;
            break;
        }
    }
    if mid_bigger {
        let mut already_bigger = false;

        for i in 0..end_idx {
            let target_c = if already_bigger {
                0
            } else {
                (target[i] - 97) as usize
            };
            let mut find = false;
            for j in target_c..26 {
                if map[j] > 0 {
                    map[j] -= 2;
                    current[i] = j as u8 + 97;
                    current[s.len() - i - 1] = j as u8 + 97;
                    if target[i] < current[i] {
                        already_bigger = true;
                    }
                    find = true;
                    break;
                }
            }
            if !find {
                break;
            }
        }
        if &current[..] > target {
            return String::from_utf8(current).unwrap();
        } else {
            return "".into();
        }
    } else {
        let mut already_bigger = false;
        for i in 0..end_idx {
            let target_c = if already_bigger {
                0
            } else {
                (target[i] - 97) as usize
            };
            let mut find = false;

            for j in target_c + 1..26 {
                if map[j] > 0 {
                    map[j] -= 2;
                    current[i] = j as u8 + 97;
                    current[s.len() - i - 1] = j as u8 + 97;
                    if current[i] > target[i] {
                        already_bigger = true
                    }
                    find = true;
                    break;
                }
            }
            if !find {
                for j in target_c..target_c + 1 {
                    if map[j] > 0 {
                        map[j] -= 2;
                        current[i] = j as u8 + 97;
                        current[s.len() - i - 1] = j as u8 + 97;
                        if current[i] > target[i] {
                            already_bigger = true
                        }
                        find = true;
                        break;
                    }
                }
            }
            if !find {
                break;
            }
        }
        if &current[..] > target {
            return String::from_utf8(current).unwrap();
        } else {
            return "".into();
        }
    }
    "".into()
}

pub fn max_product(nums: Vec<i32>) -> i64 {
    let mut nums = nums
        .into_iter()
        .map(|x| x.abs() as i64)
        .collect::<Vec<i64>>();
    nums.sort();

    let x = 1_00_000 as i64;
    (nums[nums.len() - 1] as i64 * nums[nums.len() - 2] as i64 * x).abs()
}
pub fn find_missing_elements(nums: Vec<i32>) -> Vec<i32> {
    use std::collections::HashSet;
    let mut set: HashSet<i32> = HashSet::from_iter(nums.iter().copied());
    let max = nums.iter().max().copied().unwrap();
    let min = nums.iter().min().copied().unwrap();
    let mut res = vec![];
    for i in min..max {
        if !set.contains(&i) {
            res.push(i);
        }
    }
    res
}
pub fn count_unguarded(m: i32, n: i32, guards: Vec<Vec<i32>>, walls: Vec<Vec<i32>>) -> i32 {
    #[derive(Default, Clone, Copy)]
    struct GridCell {
        up: bool,
        down: bool,
        left: bool,
        right: bool,
        blocked: bool,
        visited: bool,
        is_gurad: bool,
    }
    let mut grid = vec![vec![GridCell::default(); n as usize]; m as usize];
    for g in walls {
        grid[g[0] as usize][g[1] as usize].blocked = true;
        grid[g[0] as usize][g[1] as usize].visited = true
    }
    for g in guards {
        let mut x = g[0];
        let y = g[1];
        grid[x as usize][y as usize].is_gurad = true;
        grid[x as usize][y as usize].visited = true;

        // up
        while x >= 0 {
            if grid[x as usize][y as usize].up
                || grid[x as usize][y as usize].down
                || grid[x as usize][y as usize].blocked
            {
                break;
            }
            grid[x as usize][y as usize].up = true;
            grid[x as usize][y as usize].down = true;
            grid[x as usize][y as usize].visited = true;
            x -= 1;
        }
        //down
        let mut x = g[0];
        while x < grid.len() as i32 {
            if grid[x as usize][y as usize].down
                || grid[x as usize][y as usize].up
                || grid[x as usize][y as usize].blocked
            {
                break;
            }
            grid[x as usize][y as usize].up = true;
            grid[x as usize][y as usize].down = true;
            grid[x as usize][y as usize].visited = true;

            x += 1;
        }
        let x = g[0];
        let mut y = g[1];
        // right
        while y < grid[0].len() as i32 {
            if grid[x as usize][y as usize].left
                || grid[x as usize][y as usize].right
                || grid[x as usize][y as usize].blocked
            {
                break;
            }
            grid[x as usize][y as usize].left = true;
            grid[x as usize][y as usize].right = true;
            grid[x as usize][y as usize].visited = true;

            y += 1;
        }
        let mut y = g[1];
        // left
        while y >= 0 {
            if grid[x as usize][y as usize].left
                || grid[x as usize][y as usize].right
                || grid[x as usize][y as usize].blocked
            {
                break;
            }
            grid[x as usize][y as usize].left = true;
            grid[x as usize][y as usize].right = true;
            grid[x as usize][y as usize].visited = true;
            y -= 1;
        }
    }
    for g in grid.iter() {
        println!("{:?}", g.iter().map(|x| x.visited).collect::<Vec<_>>())
    }
    grid.into_iter().flatten().filter(|x| !x.visited).count() as _
}

pub fn modified_list(nums: Vec<i32>, mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    use std::collections::HashSet;
    let set: HashSet<i32> = HashSet::from_iter(nums.into_iter());
    let mut pre = Box::new(ListNode::new(2));
    let mut pre_ptr = &mut pre;

    while let Some(mut h) = head.take() {
        head = h.next.take();
        if !set.contains(&h.val) {
            pre_ptr.next = Some(h);
            pre_ptr = pre_ptr.next.as_mut().unwrap();
        }
    }

    pre.next
}

pub fn next_beautiful_number(n: i32) -> i32 {
    let len = n.to_string().len();
    match len {
        1 => 22,
        2 => {
            if n >= 22 {
                next_beautiful_number(100)
            } else {
                22
            }
        }
        3 => {
            // 122 333
            if n >= 333 {
                next_beautiful_number(1000)
            } else {
                for i in [122, 212, 221, 333] {
                    if i > n {
                        return i;
                    }
                }
                0
            }
        }
        4 => {
            // 13
            if n >= 4444 {
                next_beautiful_number(10000)
            } else {
                for i in [1333, 3133, 3313, 3331, 4444] {
                    if i > n {
                        return i;
                    }
                }
                0
            }
        }
        5 => {
            //14 23
            if n >= 55555 {
                next_beautiful_number(100000)
            } else {
                let mut nums = vec![];
                let mut selected = vec![false; 5];
                let mut s = String::new();
                full(&[1, 4, 4, 4, 4], &mut selected, &mut nums, &mut s);
                selected.fill(false);
                full(&[2, 2, 3, 3, 3], &mut selected, &mut nums, &mut s);
                nums.sort();
                for i in nums {
                    if i > n {
                        return i;
                    }
                }
                0
            }
        }
        6 => {
            //15 123 24
            if n >= 666666 {
                next_beautiful_number(1000000)
            } else {
                let mut nums = vec![];
                let mut selected = vec![false; 6];
                let mut s = String::new();
                full(&[1, 5, 5, 5, 5, 5], &mut selected, &mut nums, &mut s);
                selected.fill(false);
                full(&[2, 2, 4, 4, 4, 4], &mut selected, &mut nums, &mut s);
                selected.fill(false);
                full(&[2, 2, 3, 3, 3, 1], &mut selected, &mut nums, &mut s);
                nums.sort();
                for i in nums {
                    if i > n {
                        return i;
                    }
                }
                0
            }
        }
        7 => {
            // 1000000 ->1224444
            1224444
        }
        _ => {
            unreachable!()
        }
    }
}
fn full(s: &[u8], selected: &mut [bool], res: &mut Vec<i32>, current: &mut String) {
    if current.len() == s.len() {
        res.push(current.parse().unwrap());
    }
    for i in 0..s.len() {
        if !selected[i] {
            selected[i] = true;
            current.push(s[i] as char);
            full(s, selected, res, current);
            selected[i] = false;
        }
    }
}

fn bank() {
    struct Bank {
        account: Vec<i64>,
    }

    /**
     * `&self` means the method takes an immutable reference.
     * If you need a mutable reference, change it to `&mut self` instead.
     */
    impl Bank {
        fn new(balance: Vec<i64>) -> Self {
            Self { account: balance }
        }

        fn is_valaid_account(&self, account: i32) -> bool {
            if self.account.len() > account as usize - 1 {
                false
            } else {
                true
            }
        }
        fn transfer(&mut self, account1: i32, account2: i32, money: i64) -> bool {
            if self.is_valaid_account(account1) && self.is_valaid_account(account2) {
                let account1 = account1 as usize - 1;
                let account2 = account2 as usize - 1;
                if self.account[account1] >= money {
                    self.account[account1] -= money;
                    self.account[account2] += money;
                    return true;
                }
            }
            false
        }

        fn deposit(&mut self, account: i32, money: i64) -> bool {
            if self.is_valaid_account(account) {
                let account = account as usize - 1;
                self.account[account] += money;
                return true;
            }
            false
        }

        fn withdraw(&mut self, account: i32, money: i64) -> bool {
            if self.is_valaid_account(account) {
                let account = account as usize - 1;
                if self.account[account] >= money {
                    self.account[account] -= money;
                    return true;
                }
            }
            false
        }
    }
}

pub fn lex_smallest(s: String) -> String {
    use std::collections::VecDeque;
    let mut ss = VecDeque::from(s.as_bytes().to_vec());
    let mut pre = VecDeque::new();
    let mut min = ss.clone();
    for i in 1..=s.len() {
        let f = ss.pop_front().unwrap();
        pre.push_front(f);
        // a b c d
        // d c b a
        println!("{:?} {:?}", pre, ss);
        for (a, b) in pre.iter().chain(ss.iter()).zip(min.iter()) {
            if a < b {
                min = pre.iter().chain(ss.iter()).copied().collect();
                break;
            } else if a > b {
                break;
            }
        }
    }
    let mut ss = VecDeque::from(s.as_bytes().to_vec());
    let mut pre = VecDeque::new();
    for i in 1..=s.len() {
        let f = ss.pop_back().unwrap();
        pre.push_back(f);
        // a b c d
        // b c
        println!("-{:?} {:?}", pre, ss);
        //
        for (a, b) in pre.iter().chain(ss.iter()).zip(min.iter()) {
            if a < b {
                min = pre.iter().chain(ss.iter()).copied().collect();
                break;
            }
        }
    }
    String::from_utf8_lossy(min.make_contiguous()).to_string()
}

pub fn max_sum_of_squares(num: i32, mut sum: i32) -> String {
    if num * 9 < sum {
        return "".into();
    }
    let mut res = vec![0; num as usize];
    let mut i = 0;

    while sum != 0 {
        res[i] = 9.min(sum);
        sum -= res[i];
        i += 1;
    }

    String::from_utf8(res.into_iter().map(|x| (x + 48) as u8).collect()).unwrap()
}

pub fn min_operations(mut nums1: Vec<i32>, nums2: Vec<i32>) -> i64 {
    let last = nums2.last().copied().unwrap();
    let mut i = 0;
    let mut min_operations = i64::MAX;
    let mut diff = vec![0; nums1.len()];
    for i in 0..nums1.len() {
        diff[i] = (nums1[i] - nums2[i]).abs() as i64;
    }
    let mut pre_sum = vec![0; nums1.len() + 1];
    for i in 0..nums1.len() {
        pre_sum[i + 1] = pre_sum[i] + diff[i];
    }

    for (j, &n) in nums1.iter().enumerate() {
        let mut op = 1;
        if n < nums2[j] && n < last {
            op += (nums2[j].min(last) - n) as i64;
            op += ((nums2[j] - last).abs()) as i64;
            op += pre_sum[j];
            op += pre_sum[pre_sum.len() - 1] - pre_sum[j + 1];
        }
        if n > nums2[j] && n > last {
            op += (n - nums2[j].max(last)) as i64;
            op += ((nums2[j] - last).abs()) as i64;
            op += pre_sum[j];
            op += pre_sum[pre_sum.len() - 1] - pre_sum[j + 1];
        }
        if n > nums2[j] && n < last {
            //  5 2 7
            op += (last - n) as i64;
            op += (n - nums2[j]) as i64;
            op += pre_sum[j];
            op += pre_sum[pre_sum.len() - 1] - pre_sum[j + 1];
        }
        if n < nums2[j] && n > last {
            op += (n - last) as i64;
            op += (nums2[j] - n) as i64;
            op += pre_sum[j];
            op += pre_sum[pre_sum.len() - 1] - pre_sum[j + 1];
        }
        if n == last {
            op += ((n - nums2[j]).abs()) as i64;
            op += pre_sum[j];
            op += pre_sum[pre_sum.len() - 1] - pre_sum[j + 1];
        }
        if n == nums2[j] {
            op += ((n - last).abs()) as i64;
            op += pre_sum[j];
            op += pre_sum[pre_sum.len() - 1] - pre_sum[j + 1];
        }
        if (op as i64) < min_operations {
            min_operations = op as i64;
            i = j;
        }
    }
    min_operations
}

pub fn count_stable_subarrays(capacity: Vec<i32>) -> i64 {
    use std::collections::HashMap;
    let mut map: HashMap<(i32, i32), i64> = HashMap::new();
    let mut pre_sum = capacity[0];
    let mut res = 0;
    for i in 1..capacity.len() {
        let ar = capacity[i];
        let pre_sum_r_minus_1 = pre_sum + ar;
        if let Some(&x) = map.get(&(ar, pre_sum_r_minus_1)) {
            res += x;
        }
        *map.entry((capacity[i - 1], pre_sum)).or_default() += 1;
        pre_sum += ar;
    }
    res
}

pub fn num_good_subarrays(nums: Vec<i32>, k: i32) -> i64 {
    use std::collections::HashMap;
    let mut map = HashMap::from([(0, 1)]);
    let mut res = 0;
    let mut last_start = 0;
    let mut sum = 0_usize;
    for i in 0..nums.len() {
        sum = (sum + nums[i] as usize);
        if i > 0 && nums[i - 1] != nums[i] {
            let mut s = sum;
            let mut extra = i - last_start;
            for j in 0..extra {
                s -= nums[i - j] as usize;
                *map.entry(s).or_default() += 1;
            }
            last_start = i;
        }
        let target = sum % k as usize;
        res += *map.entry(target).or_default();
    }
    res
}

pub fn find_x_sum(nums: Vec<i32>, k: i32, xx: i32) -> Vec<i32> {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    for i in 0..k as usize {
        *map.entry(nums[i]).or_default() += 1
    }
    let mut res = vec![];
    let mut x: Vec<(&i32, &i32)> = map.iter().map(|x| (x.1, x.0)).collect();
    x.sort();
    let start = if (xx as usize) > x.len() {
        0
    } else {
        (x.len() - xx as usize)
    };
    res.push(x[start..x.len()].iter().map(|x| (*x.0 * *x.1)).sum::<i32>());
    for i in k as usize..nums.len() {
        *map.entry(nums[i]).or_default() += 1;
        *map.entry(nums[i - k as usize]).or_default() -= 1;

        let mut x: Vec<(&i32, &i32)> = map.iter().map(|x| (x.1, x.0)).collect();
        x.sort();
        let start = if (xx as usize) > x.len() {
            0
        } else {
            (x.len() - xx as usize)
        };
        res.push(x[start..x.len()].iter().map(|x| (*x.0 * *x.1)).sum::<i32>());
    }
    res
}
