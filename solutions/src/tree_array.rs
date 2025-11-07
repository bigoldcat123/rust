use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
    i32,
    iter::Map,
    num,
    ops::{Add, AddAssign, Sub},
    process::id,
    usize::MAX,
    vec,
};

use num_traits::{Bounded, Num, NumAssign, Zero};

pub struct TreeArray4<T> {
    nums: Vec<T>,
    tree: Vec<T>,
}

impl<T: Copy + Zero + NumAssign> TreeArray4<T> {
    ///
    /// # this tis goo!
    pub fn new(nums: Vec<T>) -> Self {
        let mut tree: Vec<T> = vec![T::zero(); nums.len() + 1];
        for (i, &n) in nums.iter().enumerate() {
            let j = i + 1;
            tree[j] += n;
            let next_j = j + Self::next_idx(j);
            if next_j < tree.len() {
                let a = tree[j];
                tree[next_j] += a;
            }
        }
        let a = 100;
        Self { tree, nums }
    }
    ///
    /// # idx is the idx in tree_array!!
    pub fn update(&mut self, mut idx: usize, value: T) {
        let mut delta = value - self.nums[idx - 1];
        self.nums[idx - 1] = value;
        while idx < self.tree.len() {
            self.tree[idx] += delta;
            idx += Self::next_idx(idx);
        }
    }
    /// +1 : delta =  1
    ///
    /// -1 : delta = -1
    pub fn update_delta(&mut self, idx: usize, delta: T) {
        let value = self.nums[idx - 1] + delta;
        self.update(idx, value)
    }
    /// # inclusive 1..=idx , 1-indexed!
    pub fn pre_sum(&self, mut idx: usize) -> T {
        let mut res = T::zero();
        while idx > 0 {
            res += self.tree[idx];
            idx -= Self::next_idx(idx);
        }
        res
    }
    pub fn pre_sum_with_module(&self, mut idx: usize, module: T) -> T {
        let mut res = T::zero();

        while idx > 0 {
            res = (res + self.tree[idx]) % module;
            idx -= Self::next_idx(idx);
        }
        res
    }
    pub fn query_with_module(&self, start: usize, end: usize, module: T) -> T {
        self.pre_sum_with_module(end, module) - self.pre_sum_with_module(start - 1, module)
    }
    // #inclusive! start..=end 1-indexed
    pub fn query(&self, start: usize, end: usize) -> T {
        self.pre_sum(end) - self.pre_sum(start - 1)
    }

    fn next_idx(idx: usize) -> usize {
        ((idx as isize) & (-(idx as isize))) as usize
    }
}

pub fn num_teams(rating: Vec<i32>) -> i32 {
    let mut pre = TreeArray3::new(vec![0; 1_000_00]);
    let mut post = TreeArray3::new(vec![0; 1_000_00]);
    pre.update_delta(rating[0] as usize, 1);
    for &n in rating.iter().skip(1) {
        post.update_delta(n as usize, 1);
    }
    let mut res = 0;
    for i in 1..rating.len() - 1 {
        let pre_less = pre.query(1, rating[i] as usize - 1);
        let pre_greater = pre.query(rating[i] as usize + 1, 1_000_00);
        let post_less = post.query(1, rating[i] as usize - 1);
        let post_greater = post.query(rating[i] as usize + 1, 1_000_00);
        pre.update_delta(rating[i] as usize, 1);
        post.update_delta(rating[i] as usize, -1);
        res += pre_less * post_greater;
        res == pre_greater * post_less;
    }
    res
}

pub struct TreeArray3 {
    nums: Vec<i32>,
    tree: Vec<i32>,
}
impl TreeArray3 {
    ///
    /// # this tis goo!
    pub fn new(nums: Vec<i32>) -> Self {
        let mut tree = vec![0; nums.len() + 1];
        for (i, &n) in nums.iter().enumerate() {
            let j = i + 1;
            tree[j] += n;
            let next_j = j + Self::next_idx(j);
            if next_j < tree.len() {
                tree[next_j] += tree[j];
            }
        }
        let a = 100;
        Self { tree, nums }
    }
    ///
    /// # idx is the idx in tree_array!!
    pub fn update(&mut self, mut idx: usize, value: i32) {
        let mut delta = value - self.nums[idx - 1];
        self.nums[idx - 1] = value;
        while idx < self.tree.len() {
            self.tree[idx] += delta;
            idx += Self::next_idx(idx);
        }
    }
    /// +1 : delta =  1
    ///
    /// -1 : delta = -1
    pub fn update_delta(&mut self, idx: usize, delta: i32) {
        let value = self.nums[idx - 1] + delta;
        self.update(idx, value)
    }
    /// # inclusive 1..=idx , 1-indexed!
    pub fn pre_sum(&self, mut idx: usize) -> i32 {
        let mut res = 0;
        idx = idx.min(self.nums.len());
        while idx > 0 {
            res += self.tree[idx];
            idx -= Self::next_idx(idx);
        }
        res
    }
    pub fn pre_sum_isize(&self, mut idx: usize) -> isize {
        let mut res = 0;
        idx = idx.min(self.nums.len());
        while idx > 0 {
            res += self.tree[idx] as isize;
            idx -= Self::next_idx(idx);
        }
        res
    }
    pub fn pre_sum_with_module(&self, mut idx: usize, module: i32) -> i32 {
        let mut res = 0;
        while idx > 0 {
            res = (res + self.tree[idx]) % module;
            idx -= Self::next_idx(idx);
        }
        res
    }
    pub fn query_with_module(&self, start: usize, end: usize, module: i32) -> i32 {
        self.pre_sum_with_module(end, module) - self.pre_sum_with_module(start - 1, module)
    }
    // #inclusive! start..=end 1-indexed
    pub fn query(&self, start: usize, end: usize) -> i32 {
        self.pre_sum(end) - self.pre_sum(start - 1)
    }
    pub fn query_isize(&self, start: usize, end: usize) -> isize {
        self.pre_sum_isize(end) - self.pre_sum_isize(start - 1)
    }
    fn next_idx(idx: usize) -> usize {
        ((idx as isize) & (-(idx as isize))) as usize
    }
}
pub struct TreeArray3Isize {
    nums: Vec<i32>,
    tree: Vec<isize>,
}
impl TreeArray3Isize {
    ///
    /// # this tis goo!
    pub fn new(nums: Vec<i32>) -> Self {
        let mut tree = vec![0; nums.len() + 1];
        for (i, &n) in nums.iter().enumerate() {
            let j = i + 1;
            tree[j] += n as isize;
            let next_j = j + Self::next_idx(j);
            if next_j < tree.len() {
                tree[next_j] += tree[j];
            }
        }
        let a = 100;
        Self { tree, nums }
    }
    ///
    /// # idx is the idx in tree_array!!
    pub fn update(&mut self, mut idx: usize, value: i32) {
        let mut delta = value - self.nums[idx - 1];
        self.nums[idx - 1] = value;
        while idx < self.tree.len() {
            self.tree[idx] += delta as isize;
            idx += Self::next_idx(idx);
        }
    }
    /// +1 : delta =  1
    ///
    /// -1 : delta = -1
    pub fn update_delta(&mut self, idx: usize, delta: i32) {
        let value = self.nums[idx - 1] + delta;
        self.update(idx, value)
    }

    pub fn pre_sum_isize(&self, mut idx: usize) -> isize {
        let mut res = 0;
        idx = idx.min(self.nums.len());
        while idx > 0 {
            res = res + self.tree[idx];
            idx -= Self::next_idx(idx);
        }
        res
    }

    pub fn query_isize(&self, start: usize, end: usize) -> isize {
        self.pre_sum_isize(end) - self.pre_sum_isize(start - 1)
    }
    fn next_idx(idx: usize) -> usize {
        ((idx as isize) & (-(idx as isize))) as usize
    }
}

pub fn get_min_swaps(num: String, k: i32) -> i32 {
    let mut num = num.chars().map(|x| x as u8).collect::<Vec<u8>>();
    let mut num_untouched = num.clone();
    let mut ans = 0;

    for _ in 0..k {
        next_permutation(&mut num);
    }
    for i in 0..num.len() {
        if num[i] != num_untouched[i] {
            for j in i + 1..num.len() {
                if num_untouched[j] == num[i] {
                    for k in (i..j).rev() {
                        num_untouched.swap(k, k + 1);
                        ans += 1;
                    }
                    break;
                }
            }
        }
    }
    ans
}

pub fn next_permutation(nums: &mut Vec<u8>) {
    use std::collections::BTreeMap;
    let mut used: BTreeMap<u8, i32> = BTreeMap::new();
    while let Some(last) = nums.pop() {
        *used.entry(last).or_default() += 1;
        if let Some((k, v)) = used.range_mut(last + 1..).next() {
            *v -= 1;
            nums.push(*k);
            break;
        }
    }
    for (k, v) in used {
        for _ in 0..v {
            nums.push(k);
        }
    }
}

pub fn number_of_pairs(nums1: Vec<i32>, nums2: Vec<i32>, diff: i32) -> i64 {
    use std::collections::{BTreeSet, HashMap};
    let diff_arr = nums1
        .into_iter()
        .zip(nums2)
        .map(|(a, b)| a - b)
        .collect::<Vec<i32>>();
    let mut res = 0;
    let mut set = BTreeSet::new();
    for &d in diff_arr.iter() {
        set.insert(d);
        set.insert(d + diff);
    }

    let mut i = 1;
    let mut map = HashMap::new();
    for n in set {
        map.insert(n, i);
        i += 1;
    }
    let mut tree_array = TreeArray3::new(vec![0; i]);
    for d in diff_arr {
        let right = map.get(&(d + diff)).copied().unwrap();
        res += tree_array.query_isize(1, right);
        let i = map.get(&d).copied().unwrap();
        tree_array.update_delta(i, 1);
    }
    res as _
}

pub fn count_range_sum(nums: Vec<i32>, lower: i32, upper: i32) -> i32 {
    use std::collections::{BTreeSet, HashMap};
    let mut set: BTreeSet<isize> = BTreeSet::new();
    let mut pre_sum = vec![0; nums.len() + 1];
    for (i, &n) in nums.iter().enumerate() {
        pre_sum[i + 1] = pre_sum[i] + n as isize;
    }
    let mut set = BTreeSet::new();
    set.insert(lower as isize);
    set.insert(pre_sum.last().copied().unwrap() - lower as isize);

    for &n in pre_sum.iter() {
        set.insert(n);
        set.insert(n - lower as isize);
        set.insert(n - upper as isize);
    }
    let mut res = 0;
    let mut map = HashMap::new();
    let mut idx = 1;
    for n in set {
        map.insert(n, idx);
        idx += 1;
    }
    let mut tree_array = TreeArray3::new(vec![0; map.len()]);

    let i = map.get(&0).copied().unwrap();
    tree_array.update_delta(i, 1);

    for &sum in &pre_sum[1..] {
        let left = map.get(&(sum - upper as isize)).copied().unwrap();
        let right = map.get(&(sum - lower as isize)).copied().unwrap();
        let idx = map.get(&sum).copied().unwrap();
        res += tree_array.query(left, right);
        tree_array.update_delta(idx, 1);
    }
    res
}

pub fn reverse_pairs2(nums: Vec<i32>) -> i32 {
    use std::collections::{BTreeSet, HashMap};

    let mut set = BTreeSet::new();
    for &n in nums.iter() {
        set.insert(n as isize);
        set.insert(n as isize * 2);
    }
    let mut map = HashMap::new();
    let mut idx = 1;
    for n in set {
        map.insert(n, idx);
        idx += 1;
    }
    let mut res = 0;
    let mut tree_array = TreeArray3::new(vec![0; idx]);
    for n in nums {
        let i = map.get(&(n as isize * 2)).copied().unwrap() + 1;
        res += tree_array.query(i, idx);
        tree_array.update_delta(map.get(&(n as isize)).copied().unwrap(), 1);
    }
    res
}

pub fn count_smaller(mut nums: Vec<i32>) -> Vec<i32> {
    struct TreeArrayIn {
        nums: Vec<i32>,
        tree: Vec<i32>,
    }
    impl TreeArrayIn {
        fn new(nums: Vec<i32>) -> Self {
            let mut tree = vec![0; nums.len() + 1];
            for i in 0..nums.len() {
                let i = i + 1;
                tree[i] += nums[i - 1];
                let next = Self::next_idx(i);
                if next < tree.len() {
                    tree[next] += tree[i];
                }
            }
            Self { tree, nums }
        }
        fn cal(&self, mut idx: usize) -> i32 {
            let mut res = 0;
            while idx > 0 {
                res += self.tree[idx];
                idx = Self::pre_idx(idx);
            }
            res
        }
        fn query(&self, start: usize, end: usize) -> i32 {
            self.cal(end) - self.cal(start - 1)
        }
        fn update(&mut self, mut idx: usize, value: i32) {
            let delta = value - self.nums[idx - 1];
            self.nums[idx - 1] = value;
            while idx < self.tree.len() {
                self.tree[idx] += delta;
                idx = Self::next_idx(idx);
            }
        }
        fn update_delta(&mut self, idx: usize, delta: i32) {
            let value = self.nums[idx - 1] + delta;
            self.update(idx, value);
        }
        fn next_idx(idx: usize) -> usize {
            idx + ((idx as isize) & (-(idx as isize))) as usize
        }
        fn pre_idx(idx: usize) -> usize {
            idx - ((idx as isize) & (-(idx as isize))) as usize
        }
    }
    let min = nums.iter().min().copied().unwrap();
    if min <= 0 {
        for n in nums.iter_mut() {
            *n += (-min + 1)
        }
    }
    let max = nums.iter().max().copied().unwrap();

    let mut tree_array = TreeArrayIn::new(vec![0; max as usize]);

    let mut res = vec![0; nums.len()];
    let len = nums.len();
    for (i, n) in nums.into_iter().rev().enumerate() {
        res[len - i - 1] = tree_array.query(1, n as usize - 1);
        tree_array.update_delta(n as usize, 1);
    }
    res
}

pub fn reverse_pairs(record: Vec<i32>) -> i32 {
    if record.is_empty() {
        return 0;
    }
    let mut res = 0;
    let mut record = record
        .into_iter()
        .enumerate()
        .collect::<Vec<(usize, i32)>>();
    record.sort_by_key(|x| x.1);
    let mut x = 1;
    let mut pre = i32::MIN;
    for i in 0..record.len() {
        let o = record[i].1;
        if record[i].1 == pre {
            record[i].1 = pre
        } else {
            record[i].1 = x;
            x += 1;
        }
        pre = o;
    }
    record.sort_by_key(|x| x.0);
    let mut tree_array = TreeArray3::new(vec![0; x as usize]);
    for (_, n) in record {
        res += tree_array.query(n as usize + 1, x as usize);
        tree_array.update_delta(n as usize, 1);
    }
    res
}
pub fn get_subarray_beauty(mut nums: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
    for n in nums.iter_mut() {
        *n += 51;
    }
    let mut tree_array = TreeArray3::new(vec![0; 103]);
    for i in 0..k as usize {
        tree_array.update_delta(nums[i] as usize, 1);
    }
    let mut res = vec![0; nums.len() - k as usize + 1];
    for i in 0..=nums.len() - k as usize {
        {
            let mut l = 1;
            let mut r = 50;
            while l <= r {
                let mid = (r - l) / 2 + l;
                if tree_array.query(1, mid) < x {
                    l = mid + 1;
                } else if tree_array.query(1, mid) > x {
                    r = mid - 1;
                } else {
                    l = mid;
                    break;
                }
            }
            if l < 51 {
                res[i] = l as i32 - 52;
            }
            // for j in 1..=50 {
            //     if tree_array.query(1, j) >= x {
            //         res[i] = j as i32 - 51;
            //         break;
            //     }
            // }
        }
        if i + (k as usize) < nums.len() {
            tree_array.update_delta(nums[i] as usize, -1);
            tree_array.update_delta(nums[i + k as usize] as usize, 1);
        }
    }
    res
}

pub fn count_operations_to_empty_array(nums: Vec<i32>) -> i64 {
    let mut res = nums.len();
    let mut sorted_idx = (0..nums.len()).collect::<Vec<_>>();
    sorted_idx.sort_by(|&a, &b| nums[a].cmp(&nums[b]));
    let mut pre = 0;
    let mut tree_array = TreeArray3::new(vec![0; nums.len()]);
    let len = nums.len();
    for next in sorted_idx {
        if next >= pre {
            res += ((next - pre) - tree_array.query(pre + 1, next + 1) as usize);
        } else {
            res += ((len - pre) + next
                - tree_array.query(pre + 1, len) as usize
                - tree_array.query(1, next + 1) as usize)
        }
        pre = next;
        tree_array.update_delta(next, 1);
    }
    res as _
}

pub fn good_triplets(mut nums1: Vec<i32>, mut nums2: Vec<i32>) -> i64 {
    let mut map = vec![0; nums1.len() + 1];
    for (i, n) in nums1.iter_mut().enumerate() {
        map[*n as usize] = i;
        *n = i as _;
    }
    for n in nums2.iter_mut() {
        *n = map[*n as usize] as _;
    }
    let mut tree_array = TreeArray4::new(vec![0; nums1.len()]);
    let mut len = nums1.len();
    let mut res = 0;
    for (i, n) in nums2.into_iter().enumerate() {
        let y = map[n as usize];
        let less = tree_array.query(1, n as usize - 1) as usize;
        res += less * (len - 1 - y - (i - less));
        tree_array.update_delta(n as usize, 1);
    }
    res as _
}

pub fn count_rectangles(mut rectangles: Vec<Vec<i32>>, mut points: Vec<Vec<i32>>) -> Vec<i32> {
    rectangles.sort_by_key(|x| x[1]);
    let mut ps = (0..points.len()).collect::<Vec<usize>>();

    ps.sort_by(|&a, &b| points[b][1].cmp(&points[a][1]));
    println!("{:?}", ps);

    let mut res = vec![0; points.len()];
    let mut rects = vec![];
    for p in ps {
        let len = rectangles.len();
        while let Some(rect) = rectangles.last() {
            if rect[1] >= points[p][1] {
                // y
                rects.push(rect[0]); // x
                rectangles.pop();
            } else {
                break;
            }
        }
        if len != rectangles.len() {
            rects.sort();
        }
        // 1 2 3 3 4 4 5 => 4
        // 0 1 2 3 4 5 6
        //

        res[p] = (rects.len()
            - match rects.binary_search(&(points[p][0] - 1)) {
                Ok(idx) => idx + 1,
                Err(idx) => idx,
            }) as _;
    }
    res
}

pub fn process_queries(queries: Vec<i32>, m: i32) -> Vec<i32> {
    let mut res = vec![];
    let mut premutations = vec![1; m as usize];
    let mut empty = vec![0; queries.len()];
    empty.append(&mut premutations);
    let mut pos = vec![0; m as usize + 1];
    for i in 1..=m as usize {
        pos[i] = queries.len() + 1 + i;
    }
    let mut tree_array = TreeArray3::new(empty);
    let mut open_idx = queries.len();
    for q in queries {
        let idx = pos[q as usize];
        res.push(tree_array.query(1, idx - 1));
        tree_array.update(idx, 0);
        tree_array.update(open_idx, 1);
        pos[q as usize] = open_idx;
        open_idx -= 1;
    }
    res
}

pub fn best_team_score(scores: Vec<i32>, ages: Vec<i32>) -> i32 {
    let mut score_age: Vec<(i32, i32)> = scores.into_iter().zip(ages).collect();
    score_age.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut dp = vec![0; score_age.len()];
    dp[0] = score_age[0].0;
    for i in 1..dp.len() {
        let mut max = 0;
        for j in 0..i {
            if score_age[j].1 == score_age[i].1 {
                max = max.max(dp[j]);
            } else if score_age[j].0 <= score_age[i].0 {
                max = max.max(dp[j]);
            }
        }
        dp[i] = max + score_age[i].0;
    }
    dp.iter().max().copied().unwrap()
}

pub fn create_sorted_array(instructions: Vec<i32>) -> i32 {
    let mut tree_array = TreeArray3::new(vec![0; 1_000_00]);
    let mut res = 0;
    for i in instructions {
        res = (res
            + tree_array
                .query_with_module(i as usize + 1, 1_000_00, 1_000_000_007)
                .min(tree_array.query_with_module(1, i as usize - 1, 1_000_000_007)))
            % 1_000_000_007;
        tree_array.update_delta(i as usize, 1);
    }
    res as _
}

fn calc_deeps(mut n: i64) -> i32 {
    let mut deep = 0;
    while n != 1 {
        n = n.count_ones() as i64;
        deep += 1;
    }
    deep
}

pub fn count_of_peaks(mut nums: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    let mut is_peek = vec![0; nums.len()];
    for i in 1..nums.len() - 1 {
        if nums[i] > nums[i - 1] && nums[i] > nums[i + 1] {
            is_peek[i] = 1;
        }
    }
    let mut tree_arry = TreeArray3::new(is_peek);
    let mut res = vec![];

    for q in queries {
        if q[0] == 1 {
            let l = q[1];
            let r = q[2];
            let mut ans = tree_arry.query(l as usize + 1, r as usize + 1);
            if tree_arry.nums[l as usize] == 1 {
                ans -= 1;
            }
            if tree_arry.nums[r as usize] == 1 {
                ans -= 1;
            }
            res.push(ans.max(0));
        } else {
            let idx = q[1] as usize;
            let value = q[2];
            nums[idx] = value;
            if idx == 0 {
                if nums[idx + 1] > nums[idx]
                    && nums[idx + 1] > nums[idx + 2]
                    && tree_arry.nums[idx + 1] == 0
                {
                    tree_arry.update_delta(idx + 1 + 1, 1);
                } else if (nums[idx + 1] <= nums[idx] || nums[idx + 1] <= nums[idx + 2])
                    && tree_arry.nums[idx + 1] != 0
                {
                    tree_arry.update_delta(idx + 1 + 1, -1);
                }
            } else if idx == nums.len() - 1 {
                if nums[idx - 1] > nums[idx]
                    && nums[idx - 1] > nums[idx - 2]
                    && tree_arry.nums[idx - 1] == 0
                {
                    tree_arry.update_delta(idx, 1);
                } else if (nums[idx - 1] <= nums[idx] || nums[idx - 1] <= nums[idx - 2])
                    && tree_arry.nums[idx - 1] != 0
                {
                    tree_arry.update_delta(idx, -1);
                }
            } else {
                if idx + 2 < nums.len() {
                    if nums[idx + 1] > nums[idx]
                        && nums[idx + 1] > nums[idx + 2]
                        && tree_arry.nums[idx + 1] == 0
                    {
                        tree_arry.update_delta(idx + 1 + 1, 1);
                    } else if (nums[idx + 1] <= nums[idx] || nums[idx + 1] <= nums[idx + 2])
                        && tree_arry.nums[idx + 1] != 0
                    {
                        tree_arry.update_delta(idx + 1 + 1, -1);
                    }
                }
                if idx >= 2 {
                    if nums[idx - 1] > nums[idx]
                        && nums[idx - 1] > nums[idx - 2]
                        && tree_arry.nums[idx - 1] == 0
                    {
                        tree_arry.update_delta(idx, 1);
                    } else if (nums[idx - 1] <= nums[idx] || nums[idx - 1] <= nums[idx - 2])
                        && tree_arry.nums[idx - 1] != 0
                    {
                        tree_arry.update_delta(idx, -1);
                    }
                }

                if nums[idx] > nums[idx - 1]
                    && nums[idx] > nums[idx + 1]
                    && tree_arry.nums[idx] == 0
                {
                    tree_arry.update_delta(idx + 1, 1);
                } else if (nums[idx] <= nums[idx - 1] || nums[idx] <= nums[idx + 1])
                    && tree_arry.nums[idx] != 0
                {
                    tree_arry.update_delta(idx + 1, -1);
                }
            }
        }
    }
    res
}
pub fn popcount_depth(mut nums: Vec<i64>, queries: Vec<Vec<i64>>) -> Vec<i32> {
    let mut trees = vec![TreeArray2::new(nums.len(), vec![]); 6];
    for (i, &n) in nums.iter().enumerate() {
        let deeps = calc_deeps(n);
        if deeps <= 5 {
            trees[deeps as usize].update(i + 1);
        }
    }

    let mut res = vec![];
    for q in queries {
        for e in trees.iter() {
            println!("{:?}", e.tree);
        }
        if q[0] == 1 {
            let l = q[1] as usize;
            let r = q[2] as usize;
            let k = q[3] as usize;
            res.push(trees[k].query(l + 1, r + 1)); // inclusive
        } else {
            let idx = q[1] as usize;
            let n = q[2];
            let deeps_prew = calc_deeps(nums[idx]);
            if deeps_prew <= 5 {
                trees[deeps_prew as usize].update_minus(idx + 1);
            }
            nums[idx] = n;
            let deeps = calc_deeps(n);

            if deeps <= 5 {
                trees[deeps as usize].update(idx + 1);
            }
        }
    }
    res
}

pub fn result_array(nums: Vec<i32>) -> Vec<i32> {
    let mut nums: Vec<(usize, usize)> = nums
        .iter()
        .enumerate()
        .map(|(x, y)| (x, *y as usize))
        .collect();
    nums.sort_by_key(|x| x.1);
    let mut map = vec![0; nums.len()];
    let mut current_num = 1;
    for i in 0..nums.len() {
        map[current_num] = nums[i].1;
        if i >= 1 && map[nums[i - 1].1] == nums[i].1 {
            nums[i].1 = nums[i - 1].1;
        } else {
            nums[i].1 = current_num;
        }
        current_num += 1;
    }
    nums.sort_by_key(|x| x.0);
    let mut tree1 = TreeArray2::new(current_num, vec![]);
    let mut tree2 = TreeArray2::new(current_num, vec![]);

    let mut arr1 = vec![nums[0].1];
    tree1.update(nums[0].1);
    let mut arr2 = vec![nums[1].1];
    tree2.update(nums[1].1);

    for i in 2..nums.len() {
        let n = nums[i].1;
        if tree1.query(n + 1, current_num) > tree2.query(n + 1, current_num) {
            arr1.push(n);
            tree1.update(n);
        } else if tree1.query(n + 1, current_num) < tree2.query(n + 1, current_num) {
            arr2.push(n);
            tree2.update(n);
        } else if arr1.len() <= arr2.len() {
            arr1.push(n);
            tree1.update(n);
        } else {
            arr2.push(n);
            tree2.update(n);
        }
    }
    let mut arr1: Vec<_> = arr1.into_iter().map(|x| map[x] as i32).collect();
    let mut arr2: Vec<_> = arr2.into_iter().map(|x| map[x] as i32).collect();

    println!("{arr1:?}");
    println!("{arr2:?}");
    arr1.append(&mut arr2);
    arr1
}

pub struct TreeArray {
    pub tree: Vec<i32>,
    pub nums: Vec<i32>,
}
impl TreeArray {
    pub fn new(nums: Vec<i32>) -> Self {
        let mut tree = vec![0; nums.len() + 1];
        for (mut i, &n) in nums.iter().enumerate() {
            let mut i = i + 1;
            tree[i] += n;
            loop {
                let next_idx = i + Self::next_indes(i);
                println!("{next_idx}");
                if next_idx < tree.len() {
                    tree[next_idx] += n;
                } else {
                    break;
                }
                i = next_idx;
            }
        }
        Self { tree, nums }
    }
    fn next_indes(idx: usize) -> usize {
        ((idx as isize) & (-(idx as isize))) as usize
    }
    fn update(&mut self, n: i32, mut idx: usize) {
        let delta = n - self.nums[idx];
        idx += 1;
        while idx < self.tree.len() {
            self.tree[idx] += delta;
            idx = Self::next_indes(idx);
        }
    }
    fn calc(&self, mut end: usize) -> i32 {
        let mut res = 0;
        while end > 0 {
            res += self.tree[end];
            end -= (end as isize & (-(end as isize))) as usize
        }
        res
    }
    fn query(&self, start: usize, end: usize) -> i32 {
        self.calc(end) - self.calc(start)
    }
}

struct NumArray {
    tree: TreeArray,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl NumArray {
    fn new(nums: Vec<i32>) -> Self {
        Self {
            tree: TreeArray::new(nums),
        }
    }

    fn update(&mut self, index: i32, val: i32) {
        self.tree.update(val, index as usize);
    }

    fn sum_range(&self, left: i32, right: i32) -> i32 {
        self.tree.query(left as usize, right as usize + 1)
    }
}
#[derive(Clone)]
struct TreeArray2 {
    tree: Vec<i32>,
    nums: Vec<i32>,
}
impl TreeArray2 {
    fn new(size: usize, nums: Vec<i32>) -> Self {
        let mut tree = vec![0; nums.len() + 1];
        for i in 0..nums.len() {
            let i = i + 1;
            tree[i] += nums[i];
            let next_i = (i as isize & (-(i as isize))) as usize;
            if next_i < tree.len() {
                tree[next_i] += tree[i];
            }
        }
        Self { tree, nums }
    }
    fn update_with_value(&mut self, mut idx: usize, value: i32) {
        let delta = value - self.nums[idx];
        self.nums[idx - 1] = value;
        while idx < self.tree.len() {
            self.tree[idx] += delta;
            idx += ((idx as isize) & (-(idx as isize))) as usize;
        }
    }
    fn update(&mut self, mut idx: usize) {
        self.nums[idx - 1] += 1;
        while idx < self.tree.len() {
            self.tree[idx] += 1;
            idx += ((idx as isize) & (-(idx as isize))) as usize;
        }
    }
    fn update_minus(&mut self, mut idx: usize) {
        self.nums[idx - 1] -= 1;

        while idx < self.tree.len() {
            self.tree[idx] -= 1;
            idx += ((idx as isize) & (-(idx as isize))) as usize;
        }
    }
    fn cal(&self, mut idx: usize) -> i32 {
        let mut res = 0;
        while idx > 0 {
            res += self.tree[idx];
            idx -= ((idx as isize) & (-(idx as isize))) as usize;
        }
        res
    }
    fn query(&self, start: usize, end: usize) -> i32 {
        self.cal(end) - self.cal(start - 1)
    }
}
