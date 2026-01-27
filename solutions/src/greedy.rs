pub fn bag_of_tokens_score(tokens: Vec<i32>, power: i32) -> i32 {
        // score -> power
        // power -> score
    }

pub fn min_operations(mut nums1: Vec<i32>,mut nums2: Vec<i32>) -> i32 {
    let max_len = nums1.len().max(nums2.len());
    let min_len = nums1.len().min(nums2.len());
    if min_len * 6 < max_len {
        return -1;
    }
    let mut diff = 0;
    for &n in nums1.iter() {
        diff += n;
    }
    for &n in nums2.iter() {
        diff += n;
    }
    if diff < 0 {
        diff = -diff;
        let mut p = nums1;
        nums1 = nums2;
        nums2 = p;
    }
    let mut cnt = vec![0;6];
    for &n in nums1.iter() {
        cnt[n as usize - 1] += 1;
    }
    for &n in nums2.iter() {
        cnt[6 - n as usize] += 1;
    }
    let mut s = 0;
    for i in (0..=5).rev(){
        if diff <= cnt[i] * i as i32 {
            return (diff + i as i32 - 1) / i as i32
        }
        diff -= cnt[i] * i as i32;
        s += cnt[i];
    }
    unreachable!()
}

pub fn min_sum_square_diff(nums1: Vec<i32>, nums2: Vec<i32>, k1: i32, k2: i32) -> i64 {
    use std::collections::BinaryHeap;
    let mut k = k1 as i64 + k2 as i64;
    let mut diff: BinaryHeap<i64> = nums1
        .into_iter()
        .zip(nums2)
        .map(|(a, b)| (a - b).abs() as i64)
        .collect();

    while let Some(mut max) = diff.peek_mut()
        && *max > 0
        && k > 0
    {
        *max -= 1;
        // k -= 1;
    }
    let mut ans = 0;
    for v in diff {
        ans += v * v;
    }
    ans
}
pub fn max_total(value: Vec<i32>, limit: Vec<i32>) -> i64 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut ans = 0;
    let mut vl: BinaryHeap<_> = value
        .into_iter()
        .zip(limit)
        .map(|(a, b)| (Reverse(b), a))
        .collect();
    let mut selected: BinaryHeap<Reverse<i32>> = BinaryHeap::new();
    while let Some((Reverse(limit), value)) = vl.pop() {
        ans += value as i64;
        selected.push(Reverse(limit));
        let mut active_number = selected.len();
        while let Some((Reverse(limit), value)) = vl.peek()
            && *limit as usize <= active_number
        {
            vl.pop();
        }
        while let Some(&Reverse(limit)) = selected.peek()
            && limit as usize <= active_number
        {
            selected.pop();
        }
    }

    ans
}

pub fn max_run_time(mut n: i32, mut batteries: Vec<i32>) -> i64 {
    let mut s = batteries.iter().map(|x| *x as i64).sum::<i64>();
    batteries.sort();
    for &b in batteries.iter().rev() {
        if b as i64 <= s / n as i64 {
            return s / n as i64;
        }
        s -= b as i64;
        n -= 1;
    }
    0
}

pub fn min_subsequence(mut nums: Vec<i32>) -> Vec<i32> {
    let mut sum = nums.iter().sum::<i32>();
    nums.sort();
    let mut m = 0;
    let mut ans = vec![];
    for &n in nums.iter().rev() {
        ans.push(n);
        m += n;
        sum -= n;
        if m > sum {
            break;
        }
    }
    ans
}

pub fn find_least_num_of_unique_ints(arr: Vec<i32>, mut k: i32) -> i32 {
    use std::collections::HashMap;
    let mut m: HashMap<i32, i32> = HashMap::new();
    for a in arr {
        *m.entry(a).or_default() += 1
    }
    let mut m: Vec<(i32, i32)> = m.into_iter().collect();
    m.sort_by_key(|x| x.1);
    let mut len = m.len();
    for (_, c) in m {
        k -= c;
        if k >= 0 {
            len -= 1;
        }
    }
    len as i32
}
pub fn largest_sum_after_k_negations(mut nums: Vec<i32>, mut k: i32) -> i32 {
    let mut ans = 0;
    nums.sort();
    for i in 0..nums.len() {
        if k > 0 && nums[i] < 0 {
            nums[i] = -nums[i];
        }
    }
    if k > 0 && k % 2 != 0 {
        nums.sort();
        nums[0] = -nums[0];
    }
    nums.iter().sum()
}
pub fn max_ice_cream(mut costs: Vec<i32>, mut coins: i32) -> i32 {
    costs.sort();
    let mut ans = 0;
    for c in costs {
        coins -= c;
        if coins >= 0 {
            ans += 1;
        } else {
            break;
        }
    }
    ans
}

pub fn maximum_bags(capacity: Vec<i32>, rocks: Vec<i32>, mut additional_rocks: i32) -> i32 {
    let mut left: Vec<i32> = capacity.iter().zip(rocks).map(|(c, r)| c - r).collect();
    left.sort();
    let mut ans = 0;

    for l in left {
        additional_rocks -= l;
        if additional_rocks >= 0 {
            ans += 1;
        }
    }
    ans
}

pub fn min_deletion(s: String, k: i32) -> i32 {
    let mut map = vec![0; 26];
    for s in s.chars().into_iter().map(|x| x as u8 as usize - 97) {
        map[s] += 1;
    }
    map = map.into_iter().filter(|&x| x > 0).collect();
    map.sort();
    let x = map.len() as i32 - k;
    if x > 0 {
        return map[..x as usize].iter().sum();
    }
    0
}

pub fn minimum_boxes(apple: Vec<i32>, mut capacity: Vec<i32>) -> i32 {
    let mut sum = apple.iter().sum::<i32>();
    let mut ans = 0;
    capacity.sort();
    let mut current_c = 0;
    for c in capacity.iter().rev() {
        current_c += c;
        ans += 1;
        if current_c >= sum {
            break;
        }
    }
    ans
}
