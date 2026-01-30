pub fn avoid_flood(rains: Vec<i32>) -> Vec<i32> {
    use std::collections::{HashSet,HashMap,VecDeque,BTreeSet};
    let mut map:HashMap<i32, VecDeque<usize>> = HashMap::new();
    let mut pre_map:HashMap<i32, usize> = HashMap::new();
    for (i,&r) in rains.iter().enumerate(){
        map.entry(r).or_default().push_back(i);
        if !pre_map.contains_key(&r){
            pre_map.insert(r, i);
        }
    }
    let mut clear_lake = vec![];
    let mut full_lake = BTreeSet::new();
    for (i,&r) in rains.iter().enumerate() {
        if r == 0 {

            if let Some((_,r)) = full_lake.pop_first() {
                clear_lake.push(r);
            }else {
                clear_lake.push(1);
            }

        }else {
            if full_lake.contains(&(pre_map.get(&r).copied().unwrap(),r)) {
                return vec![]
            }
            map.get_mut(&r).unwrap().pop_front();
            let next = *map[&r].front().unwrap_or(&usize::MAX);
            *pre_map.get_mut(&r).unwrap() = next;
            full_lake.insert((next,r));
            clear_lake.push(-1);
        }
    }

    clear_lake
}

pub fn min_increments(n: i32, cost: Vec<i32>) -> i32 {
    let mut m = cost.clone();
    let mut ans = 0;

    let max = dfs_max(1, &cost, &mut m, &mut ans);
    ans
}
fn dfs_cal(n: usize, m: &Vec<i32>, ans: &mut i32) {
    if n >= m.len() {
        return;
    }
    *ans += (m[n * 2] - m[n * 2 - 1]).abs();
    dfs_cal(n * 2, m, ans);
    dfs_cal(n * 2 + 1, m, ans);
}
fn dfs_max(n: usize, cost: &Vec<i32>, m: &mut Vec<i32>, ans: &mut i32) -> i32 {
    if n > cost.len() {
        return 0;
    }
    let c = cost[n - 1];
    let max_l = dfs_max(n * 2, cost, m, ans);
    let max_r = dfs_max(n * 2 + 1, cost, m, ans);
    let max = max_r.max(max_l);
    *ans += (max_l - max_r).abs();
    m[n - 1] = max;
    max
}

pub fn min_swaps(grid: Vec<Vec<i32>>) -> i32 {
    let mut grid = grid
        .into_iter()
        .map(|x| {
            let mut c = 0;
            for n in x.iter().rev().take_while(|&&y| y == 0) {
                c += 1;
            }
            c
        })
        .collect::<Vec<_>>();
    let mut l = grid.len() - 1;
    println!("{:?}", grid);

    for i in 0..grid.len() {
        if grid[i] >= l {
            grid[i] = grid[i].min(l);
            l -= 1;
        }
    }

    println!("{:?}", grid);
    let mut ans = 0;
    let mut l = grid.len() - 1;
    for i in 0..grid.len() {
        if grid[i] >= l {
            l -= 1;
            continue;
        }
        for j in i + 1..grid.len() {
            if grid[j] >= l {
                grid.remove(j);
                grid.insert(i, l);
                ans += j - i;
                break;
            }
        }
        if grid[i] < l {
            return -1;
        }
        println!("-> {:?}", grid);

        l -= 1;
    }
    println!("{:?}", grid);

    ans as _
}
pub fn min_deletion_size(mut strs: Vec<String>) -> i32 {
    let strs: Vec<&[u8]> = strs.iter().map(|x| x.as_bytes()).collect();
    let mut deleted = vec![false; strs[0].len()];
    let mut ans = 0;
    for j in 0..strs[0].len() {
        if deleted[j] {
            continue;
        }
        let mut ok = true;
        for i in 0..strs.len() - 1 {
            if strs[i][j] > strs[i + 1][j] {
                ans += 1;
                ok = false;
                break;
            }
            if strs[i][j] == strs[i + 1][j] {
                let mut x = 1;
                while j + x < strs[0].len()
                    && !deleted[j + x]
                    && strs[i][j + x] < strs[i + 1][j + x]
                {
                    ans += 1;
                    x += 1;
                    deleted[j + x] = true;
                }
            }
        }
        if ok {
            break;
        }
    }
    ans
}
const MAX: usize = 1_000_000_1;
static LPF: std::sync::LazyLock<[i32; MAX]> = std::sync::LazyLock::new(|| {
    let mut LPF2 = [0; MAX];
    for i in 2..MAX {
        if LPF2[i] == 0 {
            for j in (i..MAX).step_by(i) {
                if LPF2[j] == 0 {
                    LPF2[j] = i as i32;
                }
            }
        }
    }
    LPF2
});
pub fn min_operations(mut nums: Vec<i32>) -> i32 {
    // let mut LPF:[i32;MAX] = [0;MAX];
    // for i in 2..MAX {
    //     if LPF[i] == 0 {
    //         for j in (i..MAX).step_by(i) {
    //             if LPF[j] == 0 {
    //                 LPF[j] = i as i32;
    //             }
    //         }
    //     }
    // }
    let mut ans = 0;
    for i in (0..nums.len() - 1).rev() {
        if nums[i] > nums[i + 1] {
            nums[i] = LPF[nums[i] as usize];
            if nums[i] > nums[i + 1] {
                return -1;
            }
            ans += 1;
        }
    }
    ans
}

pub fn matrix_score(mut grid: Vec<Vec<i32>>) -> i32 {
    let mut ans = 0;
    for i in 0..grid.len() {
        if grid[i][0] == 0 {
            for n in grid[i].iter_mut() {
                if *n == 0 {
                    *n = 1;
                } else {
                    *n = 0;
                }
            }
        }
    }
    for j in 0..grid[0].len() {
        let mut one_cnt = 0;
        for i in 0..grid.len() {
            if grid[i][j] == 1 {
                one_cnt += 1;
            }
        }
        if one_cnt < grid.len() as i32 / 2 {
            for i in 0..grid.len() {
                if grid[i][j] == 1 {
                    grid[i][j] = 0;
                } else {
                    grid[i][j] = 1;
                }
            }
        }
    }
    let mut step = 10_i32.pow(grid[0].len() as _);
    for j in 0..grid[0].len() {
        for i in 0..grid.len() {
            ans += grid[i][j] * step;
        }
        step /= 10;
    }
    ans
}

pub fn min_moves(balance: Vec<i32>) -> i64 {
    let mut balance = balance.into_iter().map(|x| x as i64).collect::<Vec<_>>();
    let sum = balance.iter().sum::<i64>();
    if sum < 0 {
        return -1;
    }
    let mut ans = 0;
    let mut neg_idx = 0;
    for i in 0..balance.len() {
        if balance[i] < 0 {
            neg_idx = i;
            break;
        }
    }
    let mut l = neg_idx as i32 - 1;
    let mut r = neg_idx as i32 + 1;
    let len = balance.len() as i32;
    while balance[neg_idx] < 0 {
        if (neg_idx as i32 - l).abs() <= (neg_idx as i32 - r).abs() {
            // search for left
            let l_idx = cal_idx(l, len);
            ans +=
                (neg_idx as i32 - l).abs() as i64 * (-balance[neg_idx]).min(balance[l_idx]) as i64;
            balance[neg_idx] += (-balance[neg_idx]).min(balance[l_idx]);
            if balance[l_idx] >= -balance[neg_idx] {
                break;
            }
            l -= 1;
        } else {
            let r_idx = cal_idx(r, len);
            ans +=
                (neg_idx as i32 - r).abs() as i64 * (-balance[neg_idx]).min(balance[r_idx]) as i64;
            balance[neg_idx] += (-balance[neg_idx]).min(balance[r_idx]);
            if balance[r_idx] >= -balance[neg_idx] {
                break;
            }
            r += 1;
        }
    }

    ans
}

fn cal_idx(idx: i32, len: i32) -> usize {
    if idx < 0 {
        return (len + idx) as usize;
    } else if idx > len {
        return (idx - len) as usize;
    } else {
        return idx as usize;
    }
}

pub fn min_operations2(mut n: i32) -> i32 {
    let mut n_arr = vec![];
    while n != 0 {
        n_arr.push(n & 1);
        n <<= 1;
    }
    let mut l = 0;
    let mut r = 0;
    let mut ans = 0;
    while r < n_arr.len() {
        while r < n_arr.len() && n_arr[r] == 0 {
            r += 1;
        }
        l = r;
        while r < n_arr.len() && n_arr[r] == 1 {
            r += 1;
        }
        if r - l >= 2 {
            if r == n_arr.len() {
                n_arr.push(1);
                ans += 1;
            } else {
                n_arr[r] = 1;
                ans += 1;
            }
        } else {
            ans += 1;
        }
        l = r;
    }
    ans
}
pub fn minimum_buckets(mut hamsters: String) -> i32 {
    let hamsters2 = hamsters.as_bytes();
    if hamsters2.windows(3).any(|x| x == [b'H', b'H', b'H'])
        || hamsters2[..2] == [b'H', b'H']
        || hamsters2[hamsters2.len() - 2..] == [b'H', b'H']
    {
        return -1;
    }
    let mut ans = 0;
    unsafe {
        let hamsters = hamsters.as_bytes_mut();
        for i in 0..hamsters.len() {
            if i > 0 && i < hamsters.len() - 1 && hamsters[i - 1] == b'H' && hamsters[i + 1] == b'H'
            {
                hamsters[i] = b'A';
                hamsters[i - 1] = b'h';
                hamsters[i + 1] = b'h';
                ans += 1;
            }
        }
        for i in 0..hamsters.len() {
            if hamsters[i] == b'H' {
                ans += 1;
            }
        }
    }
    ans
}
pub fn max_operations(mut s: String) -> i32 {
    let mut ans = 0;
    let mut cnt = 0;
    let s = s.as_bytes();
    let mut idx = s.len() as i32 - 1;
    while idx >= 0 {
        if s[idx as usize] == b'1' {
            let mut c = 0;
            while s[idx as usize] == b'1' {
                c += 1;
                idx -= 1;
            }
            ans += c * cnt;
        } else {
            while s[idx as usize] == b'0' {
                idx -= 1;
            }
            cnt += 1;
        }
    }
    ans
}

pub fn moves_to_make_zigzag(mut nums: Vec<i32>) -> i32 {
    //even
    let mut step = 0;
    let mut nums2 = nums.clone();
    for i in (0..nums2.len()).step_by(2) {
        if i > 0 {
            step += 0.max(nums2[i - 1] - nums2[i] + 1);
        }
        if i + 1 < nums2.len() {
            step += 0.max(nums2[i + 1] - nums2[i] + 1);
            nums2[i + 1] = nums2[i] - 1;
        }
    }
    let mut step2 = 0;
    for i in (1..nums.len()).step_by(2) {
        if i > 0 {
            step2 += 0.max(nums[i - 1] - nums[i] + 1);
        }
        if i + 1 < nums.len() {
            step2 += 0.max(nums[i + 1] - nums[i] + 1);
            nums[i + 1] = nums[i] - 1;
        }
    }

    step.min(step2)
}

pub fn can_make_equal(mut nums: Vec<i32>, mut k: i32) -> bool {
    let mut a = nums.iter().filter(|&&x| x < 0).count();
    let mut b = nums.len() - a;
    if a % 2 == 0 && b % 2 == 0 {
        if a > b {
            b = 1;
        } else {
            a = 1;
        }
    }
    if a % 2 == 0 {
        // negtive
        let mut c = 0;
        for i in 0..nums.len() - 1 {
            if nums[i] == -1 {
                nums[i + 1] = nums[i + 1] * -1;
                k -= 1;
            }
            if k < 0 {
                return false;
            }
        }
        true
    } else if b % 2 == 0 {
        //positive
        let mut c = 0;
        for i in 0..nums.len() - 1 {
            if nums[i] == 1 {
                nums[i + 1] = nums[i + 1] * -1;
                k -= 1;
            }
            if k < 0 {
                return false;
            }
        }
        true
    } else {
        false
    }
}
pub fn max_array_value(nums: Vec<i32>) -> i64 {
    let mut current_max = nums[nums.len() - 1] as i64;
    for n in nums.into_iter().rev().skip(1) {
        if n as i64 <= current_max {
            current_max += n as i64;
        } else {
            current_max = n as i64;
        }
    }
    current_max
}

pub fn min_flips(target: String) -> i32 {
    let target: Vec<i32> = target.chars().map(|x| x as u8 as i32).collect();
    let mut c = 0;
    let mut ans = 0;
    for n in target {
        if (c + n) % 2 == 1 {
            ans += 1;
            c += 1;
        }
    }
    ans
}

pub fn remove_almost_equal_characters(word: String) -> i32 {
    let word = word.as_bytes();
    let mut l = 0;
    let mut r = 1;
    let mut ans = 0;
    while r < word.len() {
        while r < word.len()
            && (word[r] == word[r - 1] || word[r] == word[r] - 1 || word[r] == word[r] + 1)
        {
            r += 1;
        }
        let len = r - l;
        ans += len / 2;
        l = r;
        r += 1;
    }
    ans as _
}

pub fn min_rectangles_to_cover_points(points: Vec<Vec<i32>>, w: i32) -> i32 {
    let mut p: Vec<_> = points.iter().map(|x| x[0]).collect();
    p.sort();
    let mut ans = 0;

    let mut l = 0;
    let mut r = 0;
    while r < p.len() {
        while r < p.len() && p[l] + w >= p[r] {
            r += 1;
        }
        ans += 1;
        l = r;
    }

    ans
}

pub fn min_moves_to_seat(mut seats: Vec<i32>, mut students: Vec<i32>) -> i32 {
    seats.sort();
    students.sort();
    seats
        .into_iter()
        .zip(students)
        .map(|x| (x.0 - x.1).abs())
        .sum()
}
pub fn make_similar(mut nums: Vec<i32>, mut target: Vec<i32>) -> i64 {
    for n in nums.iter_mut() {
        if *n % 2 == 0 {
            *n = -*n;
        }
    }
    for n in target.iter_mut() {
        if *n % 2 == 0 {
            *n = -*n;
        }
    }

    let mut ans = 0;
    for i in 0..nums.len() {
        ans += nums[i] as i64 - target[i] as i64;
    }
    ans / 4
}

pub fn max_profit_assignment(difficulty: Vec<i32>, profit: Vec<i32>, worker: Vec<i32>) -> i32 {
    let mut d_p: Vec<_> = difficulty.into_iter().zip(profit).collect();
    d_p.sort_by_key(|x| x.0);
    let mut ans = 0;

    let mut max = vec![d_p[0].1; d_p.len()];
    for i in 1..d_p.len() {
        max[i] = max[i - 1].max(d_p[i].1);
    }
    for w in worker {
        let idx = match d_p.binary_search_by_key(&w, |x| x.0) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };
        if idx < max.len() {
            ans += max[idx];
        }
    }
    ans
}

pub fn advantage_count(mut nums1: Vec<i32>, mut nums2: Vec<i32>) -> Vec<i32> {
    let mut nums2: Vec<(usize, i32)> = nums2.into_iter().enumerate().collect();
    nums1.sort();
    nums2.sort_by_key(|x| x.1);
    let mut l = 0;
    let mut r = nums1.len() - 1;
    let mut ans = vec![];
    for &n in nums2.iter().rev() {
        if nums1[r] > n.1 {
            ans.push((n.0, nums1[r]));
            r -= 1;
        } else {
            ans.push((n.0, nums1[l]));
            l += 1;
        }
    }
    ans.sort_by_key(|x| x.0);
    ans.into_iter().map(|x| x.1).collect()
}
pub fn check_if_can_break(mut s1: String, mut s2: String) -> bool {
    unsafe {
        s1.as_bytes_mut().sort();
        s2.as_bytes_mut().sort();
    }
    // s1 >= s2 || s2 >= s1
    s1.as_bytes().iter().zip(s2.as_bytes()).all(|(a, b)| a >= b)
        || s1.as_bytes().iter().zip(s2.as_bytes()).all(|(a, b)| a <= b)
}
pub fn match_players_and_trainers(mut players: Vec<i32>, mut trainers: Vec<i32>) -> i32 {
    players.sort();
    trainers.sort();
    let mut t = trainers.len() - 1;
    let mut ans = 0;
    for &p in players.iter().rev() {
        if trainers[t] >= p {
            ans += 1;
            t -= 1;
        }
    }
    ans
}

pub fn max_num_of_marked_indices(mut nums: Vec<i32>) -> i32 {
    nums.sort();
    let mut l = 0;
    let mut r = (nums.len() / 2) as i32;
    while l <= r {
        let mid = (r - l) / 2 + l;
        if check(&nums, mid) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    r
}
fn check(nums: &[i32], mid: i32) -> bool {
    let mut l = 0;
    for i in nums.len() - mid as usize..nums.len() {
        if nums[l] * 2 > nums[i] {
            return false;
        }
        l += 1;
    }
    true
}

pub fn maximize_greatness(mut nums: Vec<i32>) -> i32 {
    nums.sort();
    let mut ans = 0;
    let mut l = 0;
    let mut r = nums.len() - 1;
    for i in (0..nums.len()) {
        if nums[r] <= nums[i] {
            l += 1;
        } else {
            r -= 1;
            ans += 1;
        }
    }
    ans
}

pub fn num_rescue_boats(mut people: Vec<i32>, limit: i32) -> i32 {
    people.sort();
    let mut ans = 0;
    let mut l = 0;
    let mut r = people.len() - 1;
    while l <= r {
        if l == r {
            ans += 1;
            break;
        }
        if people[l] + people[r] <= limit {
            l += 1;
            r -= 1;
        } else {
            r -= 1;
        }
        ans += 1;
    }
    ans
}
pub fn smallest_range_ii(mut nums: Vec<i32>, k: i32) -> i32 {
    nums.sort();
    let len = nums.len();
    let mut ans = nums[len - 1] - nums[0];
    let mut min = nums[0];
    let mut max = nums[len - 1];
    for i in 0..nums.len() {
        let min = nums[0].min(nums[i] - k);
        let max = nums[len].max(nums[i] + k);
        ans = ans.min((max - min).abs());
    }

    ans
}

pub fn max_distance(mut arrays: Vec<Vec<i32>>) -> i32 {
    arrays.sort_by_key(|x| x[0]);
    let mut ans = (arrays[0].last().copied().unwrap() - arrays[1][0]).abs();
    for i in 1..arrays.len() {
        ans = ans.max((arrays[i].last().copied().unwrap() - arrays[0][0]).abs());
    }
    ans
}
pub fn maximum_total_sum(mut maximum_height: Vec<i32>) -> i64 {
    maximum_height.sort();
    let mut pre = i32::MAX;
    for i in (0..maximum_height.len()) {
        if maximum_height[i] >= pre {
            maximum_height[i] = pre - 1;
        }
        pre = maximum_height[i];
        if pre < 0 {
            return -1;
        }
    }
    maximum_height.iter().map(|&x| x as i64).sum()
}

pub fn max_sum(mut grid: Vec<Vec<i32>>, limits: Vec<i32>, k: i32) -> i64 {
    for g in grid.iter_mut() {
        g.sort();
    }
    let mut arr = vec![];
    for (i, g) in grid.iter().enumerate() {
        arr.extend_from_slice(&grid[i][grid.len() - limits[i] as usize..]);
    }
    arr.sort();
    arr[arr.len() - k as usize..]
        .iter()
        .map(|&x| x as i64)
        .sum()
}
pub fn max_coins(mut piles: Vec<i32>) -> i32 {
    let mut p = piles.len() / 3;
    piles.sort();
    let mut ans = 0;
    for i in (0..piles.len()).rev().step_by(2) {
        p -= 1;
        ans += piles[i - 1];
        if p == 0 {
            break;
        }
    }
    ans
}
pub fn min_increment_for_unique(mut nums: Vec<i32>) -> i32 {
    let mut ans = 0;
    nums.sort();
    let mut pre = -1;
    for i in 0..nums.len() {
        if nums[i] <= pre {
            ans += pre + 1 - nums[i];
            nums[i] = pre + 1;
        }
        pre = nums[i];
    }
    ans
}

pub fn maximum_element_after_decrementing_and_rearranging(mut arr: Vec<i32>) -> i32 {
    arr.sort();
    for i in 1..arr.len() {
        if arr[i] > arr[i - 1] + 1 {
            arr[i] = arr[i - 1] + 1;
        }
    }
    arr.last().copied().unwrap()
}
pub fn max_alternating_sum(mut nums: Vec<i32>) -> i64 {
    let mut ans = 0;
    nums.sort();
    nums = nums.into_iter().map(|x| x.abs()).collect();
    let big = (nums.len() + 1) / 2;
    for i in 0..nums.len() - big {
        ans -= nums[i] as i64 * nums[i] as i64;
    }
    for i in nums.len() - big..nums.len() {
        ans += nums[i] as i64 * nums[i] as i64;
    }
    ans
}

pub fn min_deletions(s: String) -> i32 {
    let mut cnt = vec![0; 26];
    for c in s.chars() {
        cnt[c as u8 as usize - 97] += 1;
    }
    let mut ans = 0;

    cnt.sort();
    let mut i = cnt.len() - 1;
    let mut pre = i32::MAX;
    for i in (0..cnt.len()).rev() {
        if cnt[i] >= pre {
            ans += (cnt[i] - (pre - 1)).min(cnt[i]);
            cnt[i] = pre - 1;
        }
        pre = cnt[i];
    }
    ans
}

pub fn largest_perimeter(mut nums: Vec<i32>) -> i64 {
    nums.sort();
    let mut sum = (nums[0] + nums[1]) as i64;
    let mut ans = 0;
    for i in 2..nums.len() {
        if sum > nums[i] as i64 {
            ans = ans.max(sum);
        }
        sum += nums[i] as i64;
    }
    ans
}
pub fn maximum_even_split(final_sum: i64) -> Vec<i64> {
    if final_sum % 2 == 1 {
        return vec![];
    }
    let mut ans = vec![];
    let mut c = final_sum / 2;
    for i in 1.. {
        ans.push(i * 2);
        if c <= i {
            c -= i;
        } else {
            ans.pop();
            ans.push((i - 1 + c) * 2);
            break;
        }
        if c == 0 {
            break;
        }
    }
    ans
}
pub fn min_cost(colors: String, needed_time: Vec<i32>) -> i32 {
    let mut r = 0;
    let mut l = 0;
    let colors = colors.as_bytes();
    let mut ans = 0;
    while r < colors.len() {
        while r < colors.len() && colors[l] == colors[r] {
            r += 1;
        }
        let sum = needed_time[l..r].iter().sum::<i32>();
        let max = needed_time[l..r].iter().max().copied().unwrap();
        ans += sum - max;
    }
    ans
}

pub fn minimize_sum(mut nums: Vec<i32>) -> i32 {
    nums.sort();
    if nums.len() <= 3 {
        return 0;
    }
    let mut ans = i32::MAX;
    // 0 2
    ans = ans.min(nums[nums.len() - 1] - nums[2]);
    ans = ans.min(nums[nums.len() - 1 - 1] - nums[1]);
    ans = ans.min(nums[nums.len() - 1 - 2] - nums[0]);
    ans
}

pub fn min_difference(mut nums: Vec<i32>) -> i32 {
    nums.sort();
    if nums.len() <= 4 {
        return 0;
    }
    let mut ans = i32::MAX;
    // 3 0
    ans = ans.min(nums[nums.len() - 1] - nums[3]);
    // 1 2
    ans = ans.min(nums[nums.len() - 1 - 2] - nums[1]);
    // 2 1
    ans = ans.min(nums[nums.len() - 1 - 1] - nums[2]);
    // 0 3
    ans = ans.min(nums[nums.len() - 1 - 3] - nums[0]);
    ans
}

pub fn max_distinct_elements(mut nums: Vec<i32>, k: i32) -> i32 {
    nums.sort();
    let mut pre = i32::MIN;
    for i in 0..nums.len() {
        nums[i] = (nums[i] - k).max(pre + 1).min(nums[i] + k);
        pre = nums[i];
    }
    nums.iter().collect::<std::collections::HashSet<_>>().len() as _
}

pub fn max_weight(mut pizzas: Vec<i32>) -> i64 {
    pizzas.sort();
    let mut ans = 0;
    let mut days = pizzas.len() / 4;
    let max_day = (days + 1) / 2;
    let min_day = days - max_day;
    let mut i = pizzas.len() - 1;
    for d in 0..max_day {
        ans += pizzas[i] as i64;
        i -= 1;
    }
    for d in 0..min_day {
        ans += pizzas[i - 1] as i64;
        i -= 2;
    }
    ans as _
}
pub fn max_points(technique1: Vec<i32>, technique2: Vec<i32>, k: i32) -> i64 {
    let mut t1_t2: Vec<_> = technique1
        .into_iter()
        .zip(technique2)
        .map(|(a, b)| (a as i64, b as i64))
        .map(|(a, b)| (a, b, a - b, false))
        .collect();
    let mut ans = 0;
    t1_t2.sort_by_key(|x| x.2);
    for i in t1_t2.len() - k as usize..t1_t2.len() {
        ans += t1_t2[i].0;
    }
    for (a, b, _, _) in t1_t2[..t1_t2.len() - k as usize].iter() {
        ans += a.max(b);
    }
    ans
}

pub fn maximum_score(mut cards: Vec<i32>, mut cnt: i32) -> i32 {
    cards.sort();
    let mut sum = cards[cards.len() - cnt as usize..cards.len()].iter().sum();
    if sum % 2 == 0 {
        return sum;
    }
    let pre_a0: Vec<_> = cards[..cards.len() - cnt as usize]
        .iter()
        .filter(|&x| x % 2 == 0)
        .copied()
        .collect();
    let post_a0: Vec<_> = cards[cards.len() - cnt as usize..]
        .iter()
        .filter(|&x| x % 2 == 0)
        .copied()
        .collect();
    let pre_a1: Vec<_> = cards[..cards.len() - cnt as usize]
        .iter()
        .filter(|&x| x % 2 == 1)
        .copied()
        .collect();
    let post_a1: Vec<_> = cards[cards.len() - cnt as usize..]
        .iter()
        .filter(|&x| x % 2 == 1)
        .copied()
        .collect();
    let mut ans = 0;
    if !pre_a0.is_empty() {
        ans = sum + pre_a0[pre_a0.len() - 1] - post_a1[0];
    }
    if !post_a0.is_empty() && !pre_a1.is_empty() {
        ans = ans.max(sum - post_a0[0] + pre_a1[pre_a1.len() - 1]);
    }
    ans
}
pub fn max_sum_div_three(mut nums: Vec<i32>) -> i32 {
    let sum = nums.iter().sum::<i32>();
    if sum % 3 == 0 {
        return sum;
    }
    let mut a1 = vec![];
    let mut a2 = vec![];
    for n in nums {
        if n % 3 == 1 {
            a1.push(n);
        } else if n % 3 == 2 {
            a2.push(n);
        }
    }
    a1.sort();
    a2.sort();
    if sum % 3 == 2 {
        // -2 or -1 -1
        let mut ans = 0;
        if a1.len() > 1 {
            ans = sum - a1[0] - a1[1];
        }
        if !a2.is_empty() {
            ans = ans.max(sum - a2[0]);
        }
        return ans;
    } else {
        // -1
        let mut ans = 0;
        if !a1.is_empty() {
            ans = sum - a1[0];
        }
        if a2.len() > 1 {
            ans = ans.max(sum - a2[0] - a2[1]);
        }
        return ans;
    }
    0
}

pub fn bag_of_tokens_score(mut tokens: Vec<i32>, mut power: i32) -> i32 {
    // score -> power
    // power -> score
    tokens.sort();
    let mut score = 0;

    let mut l = 0;
    let mut r = tokens.len() - 1;
    let mut ans = 0;
    while l != r {
        if power >= tokens[l] {
            power -= tokens[l];
            l += 1;
        } else if score > 0 {
            score -= 1;
            power += tokens[r];
            r -= 1;
        }
        ans = ans.max(score);
    }
    ans
}

pub fn min_operations223232(mut nums1: Vec<i32>, mut nums2: Vec<i32>) -> i32 {
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
    let mut cnt = vec![0; 6];
    for &n in nums1.iter() {
        cnt[n as usize - 1] += 1;
    }
    for &n in nums2.iter() {
        cnt[6 - n as usize] += 1;
    }
    let mut s = 0;
    for i in (0..=5).rev() {
        if diff <= cnt[i] * i as i32 {
            return (diff + i as i32 - 1) / i as i32;
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
