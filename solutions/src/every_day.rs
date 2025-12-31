use std::{
    collections::{HashMap, HashSet},
    i32,
    io::BufRead,
};

pub fn latest_day_to_cross(row: i32, col: i32, cells: Vec<Vec<i32>>) -> i32 {
    let mut grid = vec![vec![0; col as usize]; row as usize];
    let mut l = 0;
    let mut r = cells.len() - 1;
    let mut last_time = 0;
    while l <= r {
        let mid = (r - l) / 2 + l;
        if check2(&mut grid, mid, &cells) {
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    l as i32
}
fn check2(grid: &mut Vec<Vec<i32>>, day: usize, cells: &Vec<Vec<i32>>) -> bool {
    use std::collections::{HashSet, VecDeque};
    grid.iter_mut().for_each(|x| x.fill(0));

    for i in 0..=day {
        let r = cells[i][0] as usize - 1;
        let c = cells[i][1] as usize - 1;
        grid[r][c] = 1;
    }
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..grid[0].len() {
        if grid[0][i] == 0 {
            q.push_back((0, i));
            vis.insert((0, i));
        }
    }
    let d = [(0, 1), (0, -1), (1, 0), (-1, 0)];
    while !q.is_empty() {
        for (i, j) in q.split_off(0) {
            if i == grid.len() - 1 {
                return true;
            }
            for &(di, dj) in d.iter() {
                let i = i as i32 + di;
                let j = j as i32 + dj;
                if i >= 0
                    && i < grid.len() as i32
                    && j >= 0
                    && j < grid[0].len() as i32
                    && grid[i as usize][j as usize] == 0
                    && !vis.contains(&(i as usize, j as usize))
                {
                    q.push_back((i as usize, j as usize));
                    vis.insert((i as usize, j as usize));
                }
            }
        }
    }
    false
}

pub fn num_magic_squares_inside(grid: Vec<Vec<i32>>) -> i32 {
    let mut map = vec![0; 10];
    if grid.len() < 3 || grid[0].len() < 3 {
        return 0;
    }
    let mut ans = 0;
    for i in 0..grid.len() - 3 {
        fill_map(i, &mut map, &grid);
        if condition1(&map) && condition2(i, 0, &grid) {
            ans += 1;
        }
        for j in 1..grid[0].len() - 3 {
            for last_i in 0..3 {
                if grid[i + last_i][j - 1] <= 9 {
                    map[grid[i + last_i][j - 1] as usize] -= 1;
                }
            }
            for new_i in 0..3 {
                if grid[i + new_i][j + 2] <= 9 {
                    map[grid[i + new_i][j + 2] as usize] += 1;
                }
            }
            if condition1(&map) && condition2(i, j, &grid) {
                ans += 1;
            }
        }
    }
    ans
}
fn condition2(i: usize, j: usize, grid: &Vec<Vec<i32>>) -> bool {
    let sum = grid[i].iter().skip(j).take(3).sum::<i32>();
    let sum2 = grid[i][j] + grid[i + 1][j + 1] + grid[i + 2][j + 2];
    let sum3 = grid[i][j + 2] + grid[i + 1][j + 1] + grid[i + 2][j];
    let mut s1 = 0;
    let mut s2 = 0;
    let mut s3 = 0;
    for i in i..i + 3 {
        s1 += grid[i][j];
        s2 += grid[i][j + 1];
        s3 += grid[i][j + 2];
    }
    sum == grid[i + 1].iter().skip(j).take(3).sum::<i32>()
        && sum == grid[i + 2].iter().skip(j).take(3).sum::<i32>()
        && sum == sum2
        && sum == sum3
        && s1 == s2
        && s2 == s3
        && sum == s3
}
fn condition1(map: &Vec<i32>) -> bool {
    map.iter().skip(1).all(|&x| x == 1)
}
fn fill_map(i: usize, map: &mut Vec<i32>, grid: &Vec<Vec<i32>>) {
    map.fill(0);
    for i in i..i + 3 {
        for j in 0..i + 3 {
            if grid[i][j] <= 9 {
                map[grid[i][j] as usize] += 1;
            }
        }
    }
}

pub fn pyramid_transition(bottom: String, allowed: Vec<String>) -> bool {
    use std::collections::HashMap;
    let mut map: HashMap<(u8, u8), Vec<u8>> = HashMap::new();
    let bottom = bottom.as_bytes();
    let allowed = allowed.iter().map(|s| s.as_bytes()).collect::<Vec<_>>();
    for a in allowed {
        map.entry((a[0], a[1])).or_default().push(a[2]);
    }
    dfs_search(&bottom, &map)
}
fn dfs_search(bottom: &[u8], map: &HashMap<(u8, u8), Vec<u8>>) -> bool {
    if bottom.len() == 1 {
        return true;
    }
    let mut cand: Vec<Vec<u8>> = vec![vec![]; bottom.len() - 1];
    for i in 0..bottom.len() - 1 {
        if let Some(x) = map.get(&(bottom[i], bottom[i + 1])) {
            cand[i].extend(x);
        } else {
            return false;
        }
    }
    let mut c = vec![];
    dfs_c(&cand, &mut c, 0, &mut vec![]);
    for b in c {
        if dfs_search(&b, map) {
            return true;
        }
    }
    false
}
fn dfs_c(cand: &Vec<Vec<u8>>, r: &mut Vec<Vec<u8>>, i: usize, temp: &mut Vec<u8>) {
    if i == cand.len() {
        r.push(temp.clone());
        return;
    }
    for &c in cand[i].iter() {
        temp.push(c);
        dfs_c(cand, r, i + 1, temp);
    }
}
pub fn most_booked(n: i32, mut meetings: Vec<Vec<i32>>) -> i32 {
    use std::collections::{BTreeMap, BTreeSet};
    let mut avaliable_house = BTreeSet::from_iter((0..n as usize).into_iter());
    let mut current_day = 0_usize;
    let mut house_cnt = vec![0; n as usize];
    let mut on_meeting_endtime: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    meetings.sort_by_key(|x| x[0]);

    for (i, m) in meetings.into_iter().enumerate() {
        if current_day < m[0] as usize {
            current_day = m[0] as usize
        }

        while let Some((end_time2, room)) = on_meeting_endtime.pop_first() {
            if end_time2 as usize <= current_day {
                avaliable_house.extend(room);
            } else {
                on_meeting_endtime.insert(end_time2, room);
                break;
            }
        }

        let duration = (m[1] - m[0]) as usize;
        if let Some(room) = avaliable_house.pop_first() {
            house_cnt[room] += 1;
            on_meeting_endtime
                .entry(current_day + duration)
                .or_default()
                .push(room);
        } else {
            let (end_time, room) = on_meeting_endtime.pop_first().unwrap();
            avaliable_house.extend(room);

            while let Some((end_time2, room)) = on_meeting_endtime.pop_first() {
                if end_time == end_time2 {
                    avaliable_house.extend(room);
                } else {
                    on_meeting_endtime.insert(end_time, room);
                    break;
                }
            }
            current_day = end_time;
            let h = avaliable_house.pop_first().unwrap();
            house_cnt[h] += 1;
            on_meeting_endtime
                .entry(current_day + duration)
                .or_default()
                .push(h);
        }
    }
    let mut ans = 0;
    let mut max = 0;
    for (i, &n) in house_cnt.iter().enumerate() {
        if n > max {
            max = n;
            ans = i as i32;
        }
    }
    ans
}

pub fn best_closing_time(customers: String) -> i32 {
    let customers = customers.as_bytes();
    let mut totoal_time = customers.len();
    let mut pre_sum_y = vec![0];
    for (i, n) in customers.iter().enumerate() {
        if *n == b'Y' {
            pre_sum_y[i + 1] = pre_sum_y[i] + 1;
        } else {
            pre_sum_y[i + 1] = pre_sum_y[i];
        }
    }
    let mut ans = i32::MAX;
    let mut p = i32::MAX;
    for i in 0..=customers.len() {
        let before = i;
        let after = totoal_time - i;
        let before_n = before - pre_sum_y[before];
        let after_Y = after - pre_sum_y[pre_sum_y.len() - 1] - pre_sum_y[i];
        if p as usize > before_n + after_Y {
            p = (before_n + after_Y) as i32;
            ans = i as i32;
        }
    }
    ans
}
pub fn minimum_boxes(apple: Vec<i32>, mut capacity: Vec<i32>) -> i32 {
    let mut sum = apple.iter().sum::<i32>();
    let mut ans = 0;
    capacity.sort();
    for c in capacity.iter().rev() {
        if sum <= 0 {
            break;
        }
        sum -= c;
        ans += 1;
    }
    ans
}

pub fn max_two_events(mut events: Vec<Vec<i32>>) -> i32 {
    events.sort_by(|a, b| a[0].cmp(&b[0]).then(a[1].cmp(&b[1])));
    let mut ans = 0;
    let mut max = vec![events[events.len() - 1][2]; events.len()];
    for i in (0..events.len() - 1).rev() {
        max[i] = max[i + 1].max(events[i][2]);
    }
    for i in 0..events.len() {
        let endtime = events[i][1] + 1;
        let a = match events.binary_search_by_key(&endtime, |x| x[1]) {
            Ok(i) => max[i],
            Err(i) => {
                if i < max.len() {
                    max[i]
                } else {
                    0
                }
            }
        };
        ans = ans.max(a + events[i][2]);
    }
    ans
}

// fn binart_search() {

// }

pub fn min_deletion_size(mut strs: Vec<String>) -> i32 {
    let mut len = strs[0].len();
    let mut ans = 0;
    let mut str = strs
        .iter_mut()
        .map(|mut x| unsafe { x.as_mut_vec() })
        .collect::<Vec<_>>();
    for i in 0..len {
        let mut is_sorted = true;
        let mut some_equal = false;
        for j in 1..str.len() {
            if str[j][i] < str[j - 1][i] {
                is_sorted = false;
                break;
            } else if str[j][i] == str[j - 1][i] {
                some_equal = true;
            }
        }
        if !is_sorted {
            for j in 0..str.len() {
                str[j][i] = 0;
            }
            ans += 1;
        } else if some_equal {
            let mut s = true;
            for ii in 1..str.len() {
                if &str[ii][i..] < &str[ii - 1][i..] {
                    s = false;
                    break;
                }
            }
            if s {
                break;
            }
        }
    }
    ans
}

pub fn str_str(haystack: String, needle: String) -> i32 {
    let mut hystack = haystack.as_bytes();
    let mut needle = needle.as_bytes();
    let mut next = vec![0; needle.len()];
    let mut len = 0;
    let mut i = 1;
    while i < needle.len() {
        if needle[i] == needle[len] {
            len += 1;
            next[i] = len;
            i += 1;
        } else if len == 0 {
            next[i] = len;
            i += 1;
        } else {
            len = next[len - 1];
        }
    }
    let mut i = 0;
    let mut j = 0;
    for i in 0..hystack.len() {
        while j < needle.len() && j + i < hystack.len() && needle[j] == hystack[i + j] {
            j += 1;
        }
        if j == needle.len() {
            return i as i32;
        }
        j = next[j];
    }
    -1
}

pub fn min_moves2(mut balance: Vec<i32>) -> i64 {
    let mut balance = balance.into_iter().map(|x| x as i64).collect::<Vec<i64>>();
    let sum = balance.iter().sum::<i64>();
    if sum < 0 {
        return -1;
    }
    let mut ans = 0;
    let mut neg_idx = 0;
    for i in 0..balance.len() {
        if balance[i] < 0 {
            neg_idx = i as i32;
            break;
        }
    }
    let mut l = neg_idx - 1;
    let mut r = neg_idx + 1;
    let len = balance.len() as i32;
    while balance[neg_idx as usize].is_negative() {
        if balance[cal_idx(l, len)] > 0 {
            if balance[cal_idx(l, len)] <= -balance[neg_idx as usize] {
                balance[neg_idx as usize] += balance[cal_idx(l, len)];
                ans += (neg_idx - l) as i64 * balance[cal_idx(l, len)];
                l -= 1;
            } else {
                ans += (neg_idx - l) as i64 * balance[neg_idx as usize].abs();
                break;
            }
        }
        if balance[cal_idx(r, len)] > 0 {
            if balance[cal_idx(r, len)] <= -balance[neg_idx as usize] {
                balance[neg_idx as usize] += balance[cal_idx(r, len)];
                ans += (r - neg_idx) as i64 * balance[cal_idx(r, len)];
                l -= 1;
            } else {
                ans += (r - neg_idx) as i64 * balance[neg_idx as usize].abs();
                break;
            }
        }
    }
    ans
}
fn cal_idx(idx: i32, len: i32) -> usize {
    if idx >= 0 && idx < len {
        return idx as usize;
    } else if idx < 0 {
        (len + idx) as usize
    } else {
        (idx - len) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        let a = vec![1, 2, 3, 4, 5];
        for i in -5..a.len() as i32 * 2 {
            println!("{}", a[cal_idx(i, a.len() as i32)])
        }
    }
}

pub fn reverse_words(mut s: String) -> String {
    unsafe {
        let s = s.as_mut_vec();
        let mut s = s.split_mut(|&x| x == b' ');
        let a = s.next().unwrap();
        let cal_fn = |x: &&u8| match x {
            b'a' => true,
            b'e' => true,
            b'i' => true,
            b'o' => true,
            b'u' => true,
            _ => false,
        };
        let n = a.iter().filter(cal_fn).count();
        for word in s {
            if word.iter().filter(cal_fn).count() == n {
                word.reverse();
            }
        }
    }
    s
}
pub fn minimum_moves(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut ans = 0;
    let mut q: VecDeque<((i32, i32), (i32, i32))> = VecDeque::from([((0, 0), (0, 1))]);
    let mut vis = HashSet::from([((0, 0), (0, 1))]);
    let len = grid.len() as i32;

    while !q.is_empty() {
        for (tail, head) in q.split_off(0) {
            if tail == (len - 1, len - 2) && head == (len - 1, len - 1) {
                return ans;
            }
            let (down_tail, down_head) = ((tail.0 + 1, tail.1), (head.0 + 1, head.1));
            //go down
            if down_tail.0 < len
                && down_head.0 < len
                && grid[down_head.0 as usize][down_head.1 as usize] != 1
                && grid[down_tail.0 as usize][down_tail.1 as usize] != 1
                && vis.insert((down_tail, down_head))
            {
                q.push_back((down_tail, down_head));
            }
            let (right_tail, right_head) = ((tail.0, tail.1 + 1), (head.0, head.1 + 1));
            //go right
            if right_tail.0 < len
                && right_head.0 < len
                && grid[right_head.0 as usize][right_head.1 as usize] != 1
                && grid[right_tail.0 as usize][right_tail.1 as usize] != 1
                && vis.insert((right_tail, right_head))
            {
                q.push_back((right_tail, right_head));
            }
            // horizontal
            if head.0 == tail.0 {
                let (clock_tail, clock_head) = (tail, (head.0 + 1, head.1 - 1));
                if clock_tail.0 < len
                    && clock_head.0 < len
                    && grid[down_head.0 as usize][down_head.1 as usize] != 1
                    && grid[down_tail.0 as usize][down_tail.1 as usize] != 1
                    && vis.insert((clock_tail, clock_head))
                {
                    q.push_back((clock_tail, clock_head));
                }
            } else {
                //vervical
                let (ant_clock_tail, ant_clock_head) = (tail, (head.0 - 1, head.1 + 1));
                if ant_clock_tail.0 < len
                    && ant_clock_head.0 < len
                    && grid[right_head.0 as usize][right_head.1 as usize] != 1
                    && grid[right_tail.0 as usize][right_tail.1 as usize] != 1
                    && vis.insert((ant_clock_tail, ant_clock_head))
                {
                    q.push_back((ant_clock_tail, ant_clock_head));
                }
            }
        }
        ans += 1;
    }

    -1
}

pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let len = board.len();
    let max = len * len;
    let mut reference = vec![(0, 0); max + 1];
    let mut q = VecDeque::from([1]);
    let mut vis = HashSet::from([1]);
    let mut ans = 0;
    while !q.is_empty() {
        for i in q.split_off(0) {
            if i == max {
                return ans;
            }
            for next in i + 1..=i + 6 {
                if next <= max {
                    let (i, j) = cal_ref(len, next);
                    if board[i][j] != -1 {
                        if vis.insert(board[i][j] as usize) {
                            q.push_back(board[i][j] as usize);
                        }
                    } else {
                        if vis.insert(next) {
                            q.push_back(next);
                        }
                    }
                }
            }
        }
    }
    -1
}
fn cal_ref(n: usize, number: usize) -> (usize, usize) {
    let mut line = number / n;
    if number % n == 0 {
        line -= 1;
    };
    let mut extra = number - line * n;
    // high -> low
    if line % 2 != 0 {
        extra = n - extra + 1;
    }
    (n - line - 1, extra - 1)
}
pub fn get_descent_periods(prices: Vec<i32>) -> i64 {
    let mut l = 0;
    let mut r = 0;
    let mut ans = 0;
    while r < prices.len() {
        r += 1;
        while r < prices.len() && prices[r] == prices[r - 1] - 1 {
            r += 1;
        }
        let x = r - l;
        ans += (1 + x) * x / 2;
        l = r;
    }
    ans as _
}

pub fn number_of_ways(corridor: String) -> i32 {
    let mut num_of_seats = corridor.chars().filter(|&x| x == 'S').count();
    if num_of_seats % 2 != 0 {
        return 0;
    } else {
        let corridor = corridor.as_bytes();
        let mut l = 0;
        let mut r = 0;
        let mut counted = num_of_seats;
        let mut res = 1;
        while r != corridor.len() && counted > 2 {
            // search for 2 seats
            let mut seats = 0;
            while seats < 2 {
                if corridor[r] == b'S' {
                    counted -= 1;
                    seats += 1;
                }
                r += 1;
            }
            let mut ok = 1;
            while corridor[r] != b'S' {
                r += 1;
                ok += 1;
            }
            res *= ok;
        }
        res
    }
}

pub fn validate_coupons(
    code: Vec<String>,
    business_line: Vec<String>,
    is_active: Vec<bool>,
) -> Vec<String> {
    let mut res = vec![vec![]; 4];
    for i in 0..code.len() {
        if is_valaid_code(&code[i]) && isvalaid_business_line(&business_line[i]) && is_active[i] {
            res[idx(&business_line[i])].push(code[i].to_string());
        }
    }
    res.iter_mut().for_each(|x| x.sort());
    res.into_iter().flatten().collect()
}
fn idx(line: &str) -> usize {
    match line {
        "electronics" => 0,
        "grocery" => 1,
        "pharmacy" => 2,
        "restaurant" => 3,
        _ => 1,
    }
}
fn isvalaid_business_line(line: &str) -> bool {
    match line {
        "electronics" => true,
        "grocery" => true,
        "pharmacy" => true,
        "restaurant" => true,
        _ => false,
    }
}
fn is_valaid_code(code: &str) -> bool {
    code.as_bytes()
        .iter()
        .all(|&c| c >= b'z' && c <= b'a' || c >= b'Z' && c <= b'A' || c == b'_')
}

pub fn find_rotate_steps(ring: String, key: String) -> i32 {
    use std::collections::{HashSet, VecDeque};

    let ring = ring.as_bytes();
    let key = key.as_bytes();
    let mut key_idx = 0;
    let mut q = HashSet::new(); // (step, current ring_idx)
    q.insert(clock_wise(0, ring, key[key_idx]));
    q.insert(anti_clock_wise(0, ring, key[key_idx]));
    let mut ans = key.len() as i32;

    key_idx += 1;
    while !q.is_empty() && key_idx < key.len() {
        let mut x = vec![];
        for (step, ring_idx) in q.drain() {
            let (step2, idx) = clock_wise(ring_idx, ring, key[key_idx]);
            x.push((step + step2, idx));
            let (step2, idx) = anti_clock_wise(ring_idx, ring, key[key_idx]);
            x.push((step + step2, idx));
        }
        key_idx += 1;
        q.extend(x.into_iter());
    }
    let min = q.iter().map(|x| x.0).min().unwrap();

    ans + min
}
fn clock_wise(mut current: i32, ring: &[u8], key: u8) -> (i32, i32) {
    let mut step = 0;
    while ring[current as usize] != key {
        current = (current + 1) % ring.len() as i32;
        step += 1;
    }
    (step, current)
}
fn anti_clock_wise(mut current: i32, ring: &[u8], key: u8) -> (i32, i32) {
    let mut step = 0;
    while ring[current as usize] != key {
        current = (current - 1 + ring.len() as i32) % ring.len() as i32;
        step += 1;
    }
    (step, current)
}
pub fn count_covered_buildings(n: i32, mut buildings: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashSet;
    buildings.sort_by(|a, b| a[0].cmp(&b[0]).then(a[1].cmp(&b[1])));
    let mut ans = 0;
    let mut l = 0;
    let mut r = 1;
    let mut set = HashSet::new();
    while r < buildings.len() {
        let mut v = vec![];
        while r < buildings.len() && buildings[r][0] == buildings[l][0] {
            v.push((buildings[r][0], buildings[r][1]));
        }
        v.pop();
        set.extend(v.into_iter());
        l = r;
        r = l + 1;
    }
    buildings.sort_by(|a, b| a[1].cmp(&b[1]).then(a[0].cmp(&b[0])));
    let mut ans = 0;
    let mut l = 0;
    let mut r = 1;
    let mut set2 = HashSet::new();
    while r < buildings.len() {
        let mut v = vec![];
        while r < buildings.len() && buildings[r][1] == buildings[l][1] {
            v.push((buildings[r][0], buildings[r][1]));
        }
        v.pop();
        set2.extend(v.into_iter());
        l = r;
        r = l + 1;
    }
    set.intersection(&set2).count() as _
}

pub fn count_permutations(complexity: Vec<i32>) -> i32 {
    let min = complexity[0];
    if complexity[1..].iter().all(|&x| {
        println!("{} {}", x, min);
        x < min
    }) {
        let mut res = complexity.len() - 1;
        for i in 1..complexity.len() - 1 {
            res = (res * i) % 1_000_000_007;
        }
        (res as i32).max(1)
    } else {
        0
    }
}

pub fn special_triplets(nums: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut pre_map: HashMap<i32, usize> = HashMap::new();
    let mut post_map: HashMap<i32, usize> = HashMap::new();
    for &n in &nums[1..] {
        *post_map.entry(n).or_default() += 1;
    }
    pre_map.insert(nums[0], 1);
    let mut ans = 0;

    for i in 1..nums.len() - 1 {
        *post_map.entry(nums[i]).or_default() -= 1;
        let target = nums[i] * 2;
        ans += *pre_map.entry(target).or_default() * *post_map.entry(target).or_default();
        *pre_map.entry(nums[i]).or_default() += 1;
    }
    (ans % 1000_000_007) as i32
}

pub fn count_triples(n: i32) -> i32 {
    use std::collections::HashSet;
    let mut x: Vec<i32> = (1..=n).map(|x| x * x).collect();
    let mut set: HashSet<i32> = HashSet::from_iter(x[1..].iter().copied());
    let mut ans = 0;
    for i in 1..x.len() - 1 {
        set.remove(&x[i]);
        for j in 0..i {
            let target = x[i] + x[j];
            if set.contains(&target) {
                ans += 1;
            }
        }
    }
    ans
}
pub fn count_partitions(nums: Vec<i32>) -> i32 {
    let mut post_sum = nums.iter().sum::<i32>();
    let mut ans = 0;
    let mut pre_sum = 0;
    let len = nums.len();
    for n in &nums[..len - 1] {
        pre_sum += n;
        post_sum -= n;
        if (pre_sum - post_sum) % 2 == 0 {
            ans += 1;
        }
    }
    ans
}

macro_rules! cov {
    (usize) => {
        |x| x as usize
    };
}

pub fn count_collisions(directions: String) -> i32 {
    let mut stack = vec![];
    let dir = directions.as_bytes();
    let mut ans = 0;
    for &d in dir {
        if stack.is_empty() {
            stack.push(d);
        } else {
            match d {
                b'R' => {
                    stack.push(d);
                }
                b'L' => {
                    if let Some(&l) = stack.last() {
                        if l == b'R' {
                            while let Some(&l) = stack.last() {
                                if l == b'R' {
                                    stack.pop();
                                    ans += 1;
                                }
                            }
                            ans += 1;
                            stack.push(b'S');
                        } else if l == b'S' {
                            ans += 1;
                        }
                    }
                }
                b'S' => {
                    let l = stack.last().copied().unwrap();
                    if l == b'R' {
                        while let Some(&l) = stack.last() {
                            if l == b'R' {
                                stack.pop();
                                ans += 1;
                            }
                        }
                    }
                    stack.push(d);
                }
                _ => unreachable!(),
            }
        }
    }
    ans
}

pub fn count_trapezoids(points: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashMap;
    let mut map: HashMap<i32, i32> = HashMap::new();
    for p in points {
        *map.entry(p[1]).or_default() += 1
    }
    let mut ans = 0;
    let n = map
        .into_iter()
        .map(|x| x.1)
        .map(|x| (1 + x) * (x - 1) / 2)
        .map(cov!(usize))
        .collect::<Vec<_>>();
    let mut sum = n.iter().sum::<usize>();

    for x in n {
        sum -= x;
        ans += x * sum;
    }
    (ans % 1000_000_007) as i32
}

pub fn min_subarray(nums: Vec<i32>, p: i32) -> i32 {
    use std::collections::HashMap;

    let mut sum = nums.iter().map(|&x| x as usize).sum::<usize>();
    let q = (sum % p as usize) as i32;
    if q == 0 {
        return 0;
    }
    let len = nums.len() as i32;
    let mut ans = i32::MAX;
    let mut current_sum = 0;
    let mut map = HashMap::new();
    map.insert(0, -1);
    for (i, n) in nums.into_iter().enumerate() {
        current_sum += n as usize;
        let qq = (current_sum % p as usize) as i32;
        if qq >= q {
            if let Some(pre_idx) = map.get(&(qq - q)) {
                ans = ans.min(i as i32 - pre_idx);
            }
        } else if qq < q {
            if let Some(pre_idx) = map.get(&(qq + (p - q))) {
                ans = ans.min(i as i32 - pre_idx);
            }
        }
        map.insert(qq, i as i32);
    }

    if ans >= len { -1 } else { ans }
}
pub fn max_subarray_sum(nums: Vec<i32>, k: i32) -> i64 {
    let k = k as usize;
    let mut map = vec![i64::MIN; nums.len()];
    let mut ans = i64::MIN;
    let mut sum = 0;
    for (i, n) in nums.into_iter().enumerate() {
        sum += n as i64;
        let target = k - (i % k);
        ans = ans.max(sum - map[target]);
        map[i % k] = map[i % k].min(sum);
    }
    ans
}

pub fn smallest_repunit_div_by_k(k: i32) -> i32 {
    if k % 2 == 0 {
        return -1;
    }
    let mut ans = 1;
    let mut x = 1;
    while x % k != 0 {
        ans += 1;
        x = ((x << 1) % k) + 1;
    }
    ans
}

macro_rules! 干 {
    (让 $a:ident = $e:expr) => {
        let $a = $e;
    };
}

pub fn 神奇的数字(大地瓜: Vec<i32>) -> Vec<bool> {
    let mut x = 1;
    let 一个字符串 = "阿斯顿";
    干!(让 你 = 100);

    let mut y = vec![0; 大地瓜.len() + 1];
    for i in 0..大地瓜.len() {
        y[i + 1] = (y[i] << 1 + 大地瓜[i]) % 5;
    }
    y.into_iter().skip(1).map(|x| x == 0).collect()
}
// 1 0 1 0 1
// 1
// 1 0
// 1 0 1
// 1 0 1 0
// 1 0 1 0 1

pub fn max_sum_div_three(nums: Vec<i32>) -> i32 {
    let mut map = vec![vec![i32::MAX]; 3];
    for &n in nums.iter() {
        map[(n % 3) as usize].push(n);
    }
    let mut ans = nums.into_iter().sum::<i32>();
    map.iter_mut().for_each(|x| x.sort());
    if ans % 3 == 1 {
        let mut a = (ans - map[1][0]);
        if map[2].len() > 1 {
            a = a.max(ans - map[2][0] - map[2][1]);
        }
        ans = a;
    }
    if ans % 3 == 2 {
        let mut a = ans.max(ans - map[2][0]);
        if map[1].len() > 1 {
            a = a.max(ans - map[1][0] - map[1][1]);
        }
        ans = a;
    }
    ans.max(0)
}
pub fn minimum_operations(nums: Vec<i32>) -> i32 {
    nums.into_iter().map(|x| (x % 3).min(3 - (x % 3))).sum()
}
pub fn count_palindromic_subsequence(s: String) -> i32 {
    use std::collections::HashSet;
    let s: Vec<usize> = s.chars().map(|x| x as u8 as usize - 97).collect();
    let mut set = HashSet::new();
    let mut pre_sum = vec![vec![0; 26]];
    for &i in s.iter() {
        let mut pre = pre_sum.last().cloned().unwrap();
        pre[i] += 1;
        pre_sum.push(pre);
    }
    let mut res = 0;
    for &i in &s[1..s.len() - 1] {
        let pre = pre_sum[i]
            .iter()
            .zip(pre_sum[0].iter())
            .map(|(&a, b)| a - b);
        let post = pre_sum
            .last()
            .unwrap()
            .iter()
            .zip(pre_sum[i].iter())
            .map(|(&a, b)| a - b);
        for (l, r) in pre
            .enumerate()
            .zip(post.enumerate())
            .filter(|(a, b)| a.1 > 0 && b.1 > 0)
        {
            if set.insert((l.0, i, r.0)) {
                res += 1;
            }
        }
    }
    println!("{:?}", set);
    res as _
}

pub fn find_final_value(nums: Vec<i32>, mut original: i32) -> i32 {
    use std::collections::HashSet;
    let set: HashSet<i32> = HashSet::from_iter(nums.into_iter());
    while set.contains(&original) {
        original *= 2;
    }
    original
}
pub fn is_one_bit_character(bits: Vec<i32>) -> bool {
    if bits.last().copied().unwrap() != 0 {
        return false;
    } else {
        let bits = &bits[..bits.len() - 1];
        let mut i = 0;
        while i < bits.len() {
            if bits[i] == 1 {
                i += 2;
            } else {
                i += 1;
            }
        }
        i == bits.len()
    }
}
pub fn k_length_apart(nums: Vec<i32>, k: i32) -> bool {
    let mut l = 0;
    let mut r = 0;
    while r < nums.len() {
        while l < nums.len() && nums[l] == 1 {
            l += 1;
        }
        r = l;
        while r < nums.len() && nums[r] == 0 {
            r += 1;
        }
        let len = r - l;
        if r < nums.len() && len < k as _ {
            return false;
        }
    }
    true
}
pub fn num_sub(s: String) -> i32 {
    let mut s = s.as_bytes();
    let mut ans = 0;
    let mut l = 0;
    let mut r = l;
    while r < s.len() {
        while r < s.len() && s[r] == b'0' {
            r += 1;
        }
        l = r;
        while r < s.len() && s[r] == b'1' {
            r += 1;
        }
        let len = (r - l);
        ans += (1 + len) * len / 2;
    }
    ans as i32
}

pub fn range_add_queries(n: i32, queries: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut res = vec![vec![0; n as usize]; n as usize];
    for q in queries {
        let x1 = q[0] as usize;
        let y1 = q[1] as usize;
        let x2 = q[2] as usize;
        let y2 = q[3] as usize;
        for i in x1..=x2 {
            res[i][y1] += 1;
            if y2 + 1 < res[0].len() {
                res[i][y2 + 1] -= 1;
            }
        }
    }
    for i in 0..res.len() {
        for j in 1..res[0].len() {
            res[i][j] = res[i][j] + res[i][j - 1];
        }
    }
    res
}

pub fn max_operations(s: String) -> i32 {
    let s = s.as_bytes();
    let mut ans = 0;
    let mut pre_one_count = 0;
    for i in 0..s.len() - 1 {
        if s[i] == b'1' {
            pre_one_count += 1;
        }
        if s[i + 1] == b'0' && s[i] == b'1' {
            ans += pre_one_count;
        }
    }
    ans
}
/// inclusive
pub fn give_me_random_array(len: usize, max: i32, min: i32) -> Vec<i32> {
    let mut res = vec![0_i32; len];
    for i in 0..len {
        let r: i32 = rand::random_range(min..=max);
        res[i] = r;
    }
    return res;
}

pub fn min_operations(mut nums: Vec<i32>) -> i32 {
    if nums.contains(&1) {
        return nums.len() as i32 - 1;
    }
    for i in 0..nums.len() - 1 {
        let gcd_ = gcd(nums[i], nums[i + 1]);
        if gcd_ == 1 {
            return nums.len() as i32;
        }
    }
    -1
}
pub fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let p = b;
        b = a % b;
        a = p;
    }
    a
}

pub fn min_flips(mat: Vec<Vec<i32>>) -> i32 {
    let mut used = vec![];
    let mut q = vec![mat];
    let mut ans = 0;
    while !q.is_empty() {
        let pq = q.split_off(0);
        for p in pq {
            used.push(p.clone());
            if p.iter().flatten().sum::<i32>() == 0 {
                return ans;
            }
            for i in 0..p.len() {
                for j in 0..p[0].len() {
                    let mut candidate = p.clone();
                    candidate[i][j] = (candidate[i][j] + 1) % 2;
                    let i = i as i32;
                    let j = j as i32;
                    let des = [(i - 1, j), (i + 1, j), (i, j + 1), (i, j - 1)];
                    for (i, j) in des {
                        if i >= 0
                            && j >= 0
                            && (i as usize) < candidate.len()
                            && (j as usize) < candidate[0].len()
                        {
                            candidate[i as usize][j as usize] =
                                (candidate[i as usize][j as usize] + 1) % 2;
                        }
                    }
                    if !used.contains(&candidate) {
                        q.push(candidate);
                    }
                }
            }
            ans += 1;
        }
    }
    -1
}

pub fn min_mutation(start_gene: String, end_gene: String, bank: Vec<String>) -> i32 {
    let mut used = vec![];
    let mut q = vec![&start_gene];
    let mut ans = 0;
    while !q.is_empty() {
        let mut pq = q.split_off(0);
        for c in pq {
            if c == &end_gene {
                return ans;
            }
            used.push(c);
            for b in bank.iter() {
                if !used.contains(&b) && can_mutate(c, b) {
                    q.push(b);
                }
            }
        }
        ans += 1;
    }
    -1
}
fn can_mutate(from: &str, to: &str) -> bool {
    from.chars().zip(to.chars()).filter(|(x, y)| x != y).count() == 1
}

pub fn ball_game(num: i32, plate: Vec<String>) -> Vec<Vec<i32>> {
    use std::collections::HashSet;
    let plate: Vec<&[u8]> = plate.iter().map(|x| x.as_bytes()).collect();
    let mut ans = vec![];
    for j in 1..plate[0].len() - 1 {
        // up down
        if plate[0][j] == b'.'
            && dfs_ball_game(0, j as i32, &plate, (1, 0), &mut HashSet::new(), num)
        {
            ans.push(vec![0 as i32, j as i32]);
        }
        if plate[plate.len() - 1][j] == b'.'
            && dfs_ball_game(
                plate.len() as i32 - 1,
                j as i32,
                &plate,
                (-1, 0),
                &mut HashSet::new(),
                num,
            )
        {
            ans.push(vec![plate.len() as i32 - 1, j as i32]);
        }
    }
    for i in 1..plate[0].len() - 1 {
        // left right
        if plate[i][plate[0].len() - 1] == b'.'
            && dfs_ball_game(
                i as i32,
                plate[0].len() as i32 - 1,
                &plate,
                (0, -1),
                &mut HashSet::new(),
                num,
            )
        {
            ans.push(vec![i as i32, plate[0].len() as i32 - 1]);
        }
        if plate[i][0] == b'.'
            && dfs_ball_game(i as i32, 0, &plate, (0, 1), &mut HashSet::new(), num)
        {
            ans.push(vec![i as i32, 0]);
        }
    }
    ans
}
fn dfs_ball_game(
    i: i32,
    j: i32,
    plate: &Vec<&[u8]>,
    step: (i32, i32),
    visited: &mut HashSet<(usize, usize)>,
    steps: i32,
) -> bool {
    if i >= 0 && j >= 0 && (i as usize) < plate.len() && (j as usize) < plate[0].len() {
        let iusize = i as usize;
        let jusize = j as usize;
        if !visited.insert((iusize, jusize)) || steps < 0 {
            return false;
        }
        match plate[iusize][jusize] as char {
            'E' => {
                let step = turn_dir(step, 'E');
            }
            'W' => {
                let step = turn_dir(step, 'W');
            }
            'O' => return true,
            _ => {}
        }
        return dfs_ball_game(i + step.0, j + step.1, plate, step, visited, steps - 1);
    } else {
        false
    }
}
fn turn_dir(orign_step: (i32, i32), signal: char) -> (i32, i32) {
    match orign_step {
        (0, 1) => {
            if signal == 'E' {
                (1, 0)
            } else {
                (-1, 0)
            }
        } //go right
        (1, 0) => {
            if signal == 'E' {
                (0, -1)
            } else {
                (0, 1)
            }
        } //go down
        (-1, 0) => {
            if signal == 'E' {
                (0, 1)
            } else {
                (0, -1)
            }
        } //go up
        (0, -1) => {
            if signal == 'E' {
                (-1, 0)
            } else {
                (1, 0)
            }
        } //go left
        (_, _) => unreachable!(),
    }
}

pub fn find_max_form(strs: Vec<String>, m: i32, n: i32) -> i32 {
    let mut dp = vec![vec![vec![0; m as usize + 1]; n as usize + 1]; strs.len() + 1]; // i j k => current str , n (one), m(zero)
    let strs: Vec<(usize, usize)> = strs
        .into_iter()
        .map(|x| {
            let zeroes = x.chars().filter(|&x| x == '0').count();
            (zeroes, x.len() - zeroes)
        })
        .collect();
    for i in 1..=strs.len() {
        for o in 0..=n as usize {
            for z in 0..=m as usize {
                dp[i][o][z] = dp[i - 1][o][z];
                if strs[i - 1].0 <= z && strs[i - 1].1 <= o {
                    dp[i][o][z] = dp[i][o][z].max(dp[i - 1][o - strs[i - 1].1][z - strs[i - 1].0]);
                }
            }
        }
    }
    dp.last()
        .map(|x| x.last().map(|x| x.last().unwrap()).unwrap())
        .copied()
        .unwrap()
}

use crate::ListNode;

// struct SegTree {
//     tree: Vec<i64>,
// }
// impl SegTree {
//     fn new(size:usize) -> Self {
//         let mut tree = vec![0;size * 4 + 7];
//         Self { tree }
//     }
//     /// 0-indexed
//     fn update_delta(&mut self,idx:usize,delta:i64) {
//         self.update_delta_dfs(idx + 1, delta, 1, self.tree.len() - 1, 1);
//     }
//     fn update_delta_dfs(&mut self,idx:usize,delta:i64,l:usize,r:usize,tree_idx:usize) {
//         if l == r {
//             self.tree[tree_idx] += 1;
//             return;
//         }
//         let mid = (l + r) / 2;
//         if idx <= mid {
//             self.update_delta_dfs(idx, delta, l, mid, tree_idx * 2);
//         }else {
//             self.update_delta_dfs(idx, delta, mid + 1, r, tree_idx * 2 + 1);
//         }
//         self.tree[tree_idx] = self.tree[tree_idx * 2] + self.tree[tree_idx * 2 + 1]
//     }
//     /// 0-indexed inclusive
//     fn query(&self,start:usize,end:usize) -> i64{
//         self.query_dfs(start + 1, end + 1, 1, self.tree.len() - 1, 1)
//     }
//     fn query_dfs(&self,start:usize,end:usize,l:usize,r:usize,tree_idx:usize) -> i64{
//         if start == l && end == r {
//             return self.tree[tree_idx];
//         }
//         let mid = (l + r) /  2;
//         if end <= mid {
//             return self.query_dfs(start, end, l, mid, tree_idx * 2);
//         }else if start > mid {
//             return self.query_dfs(start, end, mid + 1, r, tree_idx * 2 + 1);
//         }else {
//             return self.query_dfs(start, mid, l, mid, tree_idx * 2) + self.query_dfs(mid + 1, end, mid + 1, r, tree_idx * 2 + 1);
//         }
//     }
// }
//
//
pub struct MySegTree {
    pub nodes: Vec<i64>,
    tree: Vec<i64>,
}
impl MySegTree {
    pub fn new(nodes: Vec<i64>) -> Self {
        let len = nodes.len();
        let mut tree = vec![0; len * 4 + 7];
        let mut seg_tree = Self { nodes, tree };
        seg_tree.build(1, len, 1);
        seg_tree
    }
    fn build(&mut self, l: usize, r: usize, idx: usize) {
        if l == r {
            self.tree[idx] = self.nodes[l - 1];
        } else {
            let mid = (r + l) / 2;
            self.build(l, mid, idx * 2);
            self.build(mid + 1, r, idx * 2 + 1);
            self.tree[idx] = self.tree[idx * 2] + (self.tree[idx * 2 + 1]);
        }
    }
    /// 1-indexed
    pub fn update_delta(&mut self, idx: usize, delta: i64) {
        let value = self.nodes[idx - 1] + delta;
        self.update(idx, value);
    }
    /// 1-indexed
    pub fn update(&mut self, idx: usize, value: i64) {
        self.nodes[idx - 1] = value;
        self.update_dfs(idx, 1, self.nodes.len(), value, 1);
    }
    fn update_dfs(&mut self, idx: usize, l: usize, r: usize, value: i64, tree_idx: usize) {
        if l == r {
            self.tree[tree_idx] = value;
        } else {
            let mid = (l + r) / 2;
            if idx <= mid {
                self.update_dfs(idx, l, mid, value, tree_idx * 2);
            } else {
                self.update_dfs(idx, mid + 1, r, value, tree_idx * 2 + 1);
            }
            self.tree[tree_idx] = self.tree[tree_idx * 2] + (self.tree[tree_idx * 2 + 1]);
        }
    }
    fn query_dfs(&self, ql: usize, qr: usize, l: usize, r: usize, tree_idx: usize) -> i64 {
        if ql == l && qr == r {
            return self.tree[tree_idx];
        }
        let mid = (l + r) / 2;
        if ql > mid {
            return self.query_dfs(ql, qr, mid + 1, r, tree_idx * 2 + 1);
        }
        if qr <= mid {
            return self.query_dfs(ql, qr, l, mid, tree_idx * 2);
        }
        self.query_dfs(ql, mid, l, mid, tree_idx * 2)
            + (self.query_dfs(mid + 1, qr, mid + 1, r, tree_idx * 2 + 1))
    }
    /// 1-indexed
    pub fn query(&self, ql: usize, qr: usize) -> i64 {
        self.query_dfs(ql, qr, 1, self.nodes.len(), 1)
    }
}

pub struct TreeArray3 {
    nums: Vec<i64>,
    tree: Vec<i64>,
}
impl TreeArray3 {
    ///
    /// # this tis goo!
    pub fn new(nums: Vec<i64>) -> Self {
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
    pub fn update(&mut self, mut idx: usize, value: i64) {
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
    pub fn update_delta(&mut self, idx: usize, delta: i64) {
        let value = self.nums[idx - 1] + delta;
        self.update(idx, value)
    }
    /// # inclusive 1..=idx , 1-indexed!
    pub fn pre_sum(&self, mut idx: usize) -> i64 {
        let mut res = 0;
        idx = idx.min(self.nums.len());
        while idx > 0 {
            res += self.tree[idx];
            idx -= Self::next_idx(idx);
        }
        res
    }

    // #inclusive! start..=end 1-indexed
    pub fn query(&self, start: usize, end: usize) -> i64 {
        self.pre_sum(end) - self.pre_sum(start - 1)
    }

    fn next_idx(idx: usize) -> usize {
        ((idx as isize) & (-(idx as isize))) as usize
    }
}

pub fn min_operations2(nums: Vec<i32>) -> i32 {
    let mut s: Vec<i32> = vec![];
    let mut ans = 0;
    for n in nums {
        while s.last().map_or(false, |&x| x > n) {
            s.pop();
        }
        if n == 0 {
            continue;
        }
        if s.last().map_or(true, |&x| x < n) {
            s.push(n);
            ans += 1;
        }
    }
    ans
}

pub fn max_path_score(grid: Vec<Vec<i32>>, k: i32) -> i32 {
    let cost = if grid.last().unwrap().last().copied().unwrap() > 0 {
        1
    } else {
        0
    };
    let mut min_cost = vec![vec![-1; grid[cost].len()]; grid.len()];
    for i in (0..grid.len() - 1).rev() {
        let cost = if grid[i][grid[0].len() - 1] > 0 { 1 } else { 0 };
        min_cost[i][grid[0].len() - 1] = min_cost[i + 1][grid[0].len() - 1] + cost;
    }
    for i in (0..grid[0].len() - 1).rev() {
        let cost = if grid[grid[0].len() - 1][i] > 0 { 1 } else { 0 };
        min_cost[grid[0].len() - 1][i] = min_cost[grid[0].len() - 1][i + 1] + cost;
    }
    for i in (0..grid.len() - 1).rev() {
        for j in (0..grid[0].len() - 1).rev() {
            let cost = if grid[i][j] > 0 { 1 } else { 0 };
            min_cost[i][j] = min_cost[i][j + 1].min(min_cost[i + 1][j]) + cost;
        }
    }
    let mut res = -1;
    dfs_max_path_score(0, 0, &grid, k, &mut res, 0, &min_cost);

    res
}

fn dfs_max_path_score(
    i: i32,
    j: i32,
    grid: &Vec<Vec<i32>>,
    k: i32,
    res: &mut i32,
    current_score: i32,
    min: &Vec<Vec<i32>>,
) {
    if i >= 0 && j >= 0 && (i as usize) < grid.len() && (j as usize) < grid[0].len() && k >= 0 {
        let cost = if grid[i as usize][j as usize] > 0 {
            1
        } else {
            0
        };
        let score = grid[i as usize][j as usize];
        if i as usize == grid.len() - 1 && j as usize == grid[0].len() - 1 && k >= cost {
            *res = (*res).max(current_score + score);
            return;
        }
        if min[i as usize][j as usize] > k {
            return;
        }
        dfs_max_path_score(i + 1, j, grid, k - cost, res, current_score + score, min);
        dfs_max_path_score(i, j + 1, grid, k - cost, res, current_score + score, min);
    }
}

pub fn minimum_distance(nums: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut m: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, n) in nums.into_iter().enumerate() {
        m.entry(n).or_default().push(i);
    }
    let mut min = i32::MAX;

    for (_, idxs) in m {
        if idxs.len() >= 3 {
            for i in 2..idxs.len() {
                let value =
                    (idxs[i] - idxs[i - 1]) + (idxs[i] - idxs[i - 2]) + (idxs[i - 1] - idxs[i - 2]);
                min = min.min(value as i32);
            }
        }
    }

    if min == i32::MAX { -1 } else { min }
}

pub fn count_majority_subarrays2(mut nums: Vec<i32>, target: i32) -> i64 {
    nums.iter_mut().for_each(|x| {
        if *x != target {
            *x = -1;
        } else {
            *x = 1;
        }
    });
    let mut ans = 0;
    let mut s_tree = TreeArray3::new(vec![0; nums.len() * 2]);
    s_tree.update_delta(nums.len(), 1);
    let len = nums.len();
    let mut sum = 0;
    for n in nums {
        sum += n;
        ans += s_tree.query(0, (sum - 1 + len as i32) as usize);
        s_tree.update_delta((sum + len as i32) as usize, 1);
    }
    ans
}

pub fn longest_subarray(mut nums: Vec<i32>) -> i32 {
    nums.insert(0, i32::MIN);
    let mut l = 0;
    let mut r = 1;
    let mut op = true;
    let mut ans = 0;
    let mut break_point = 0;
    let mut break_point_value = 0;
    while r < nums.len() {
        while r < nums.len() && nums[r] >= nums[r - 1] {
            r += 1;
        }
        if r < nums.len() && op {
            break_point = r;
            break_point_value = nums[r];
            op = false;
            nums[r] = nums[r - 1];
        } else {
            ans = ans.max(r - l);
            op = true;
            nums[break_point] = break_point_value;
            l = break_point;
        }
    }
    nums.reverse();
    let mut l = 0;
    let mut r = 1;
    let mut op = true;
    while r < nums.len() {
        while r < nums.len() && nums[r] <= nums[r - 1] {
            r += 1;
        }
        if r < nums.len() && op {
            break_point = r;
            break_point_value = nums[r];
            op = false;
            nums[r] = nums[r - 1];
        } else {
            ans = ans.max(r - l);
            op = true;
            nums[break_point] = break_point_value;
            l = break_point;
        }
    }

    ans as _
}

pub fn count_majority_subarrays(nums: Vec<i32>, target: i32) -> i32 {
    let mut pre_sum = vec![0; nums.len() + 1];
    for (i, &n) in nums.iter().enumerate() {
        if n == target {
            pre_sum[i] = pre_sum[i - 1] + 1;
        } else {
            pre_sum[i] = pre_sum[i - 1];
        }
    }
    let mut ans = 0;
    for i in 1..=nums.len() {
        for j in 0..i {
            let len = i - j;
            let need = len / 2;
            if pre_sum[i] - pre_sum[j] > need {
                ans += 1;
            }
        }
    }
    ans
}

pub fn min_moves(nums: Vec<i32>) -> i32 {
    let max = nums.iter().max().copied().unwrap();
    let mut ans = 0;
    for n in nums {
        ans += max - n;
    }
    ans
}

pub fn count_operations(mut num1: i32, mut num2: i32) -> i32 {
    let mut ans = 0;
    while num1 != 0 && num2 != 0 {
        if num1 > num2 {
            let times = num1 / num2;
            num1 = num1 % num2;
            ans += times;
        } else {
            let times = num2 / num1;
            num2 = num2 % num1;
            ans += times;
        }
    }
    ans
}
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
        if check(diff.as_ref(), mid, r as usize, k as i64) {
            min = mid + 1;
        } else {
            max = mid - 1;
        }
    }
    max
}
fn check(diff: &[i64], target: i64, radius: usize, mut k: i64) -> bool {
    let mut d = vec![0; diff.len()];
    let mut current = 0;
    for i in 0..diff.len() - 1 {
        current += diff[i] + d[i];
        let need = target - current;
        if need > 0 {
            if k >= need {
                current = target;
                d[i + 2 * radius + 1] -= need;
                k -= need;
            } else {
                return false;
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
        } else if let Some(mut e) = max_accurance_nums_map.first_entry() {
            if *e.key() < v {
                sum -= *e.key() * e.get().first().copied().unwrap();
                unsued_accurance_nums_map
                    .entry(*e.key())
                    .or_default()
                    .insert(e.get_mut().pop_first().unwrap());
                if e.get().is_empty() {
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
                    if e.get().is_empty() {
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
                if max_occ.is_empty() {
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
                if max_occ.is_empty() {
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
                if mac_occ.is_empty() {
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
                if mac_occ.is_empty() {
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
            if last.get().is_empty() {
                unsued_accurance_nums_map.pop_last();
            }
        } else if max_minus || unsed_add {
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
                            if e_min_max.get().is_empty() {
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
                            if e_min_max.get().is_empty() {
                                unsued_accurance_nums_map.pop_last();
                            }
                            unsued_accurance_nums_map
                                .entry(k)
                                .or_default()
                                .insert(min_one);
                        }
                    } else if e_max_min.key() == e_min_max.key()
                        && e_max_min.get().first().unwrap() < e_min_max.get().last().unwrap()
                    {
                        let a = e_max_min.get_mut().pop_first().unwrap();
                        let b = e_min_max.get_mut().pop_last().unwrap();
                        sum -= a * *e_max_min.key();
                        sum += b * *e_max_min.key();
                        e_max_min.get_mut().insert(b);
                        e_min_max.get_mut().insert(a);
                        if e_min_max.get().is_empty() {
                            unsued_accurance_nums_map.pop_last();
                        }
                        if e_max_min.get().is_empty() {
                            max_accurance_nums_map.pop_first();
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

    let x = 1_00_000_i64;
    (nums[nums.len() - 1] * nums[nums.len() - 2] * x).abs()
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
    let set: HashSet<i32> = HashSet::from_iter(nums);
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
            self.account.len() < account as usize
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

pub fn min_operations22(mut nums1: Vec<i32>, nums2: Vec<i32>) -> i64 {
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
        if op < min_operations {
            min_operations = op;
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
        sum += nums[i] as usize;
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
