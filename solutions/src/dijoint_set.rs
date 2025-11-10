use crate::TreeNode;
use std::cell::{self, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;
use std::iter::Map;
use std::mem::needs_drop;
use std::os::unix::raw::gid_t;
use std::process::id;
use std::rc::Rc;
use std::{num, vec};

struct DisjointSet {
    fa: Vec<usize>,
    size: Vec<usize>,
    cc: usize,
    value: Vec<i64>,
}
impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            fa: (0..size).collect(),
            size: vec![1; size],
            cc: size,
            value: vec![0; size],
        }
    }
    fn find(&mut self, n: usize) -> usize {
        if self.fa[n] != n {
            self.fa[n] = self.find(self.fa[n]);
        }
        self.fa[n]
    }
    fn union(&mut self, from: usize, to: usize) -> bool {
        let a = self.find(from);
        let b = self.find(to);
        if a == b {
            return false;
        }
        self.size[b] += self.size[a];
        self.fa[a] = b;
        self.value[b] += self.value[a];
        self.cc -= 1;
        true
    }
    fn get_size(&mut self, n: usize) -> usize {
        let idx = self.find(n);
        self.size[idx]
    }
    fn get_value(&mut self, n: usize) -> i64 {
        let idx = self.find(n);
        self.value[idx]
    }
}

struct DisjointSet2D {
    fa: HashMap<(i32, i32), (i32, i32)>,
    cc: usize,
}
impl DisjointSet2D {
    fn new(row: usize, colum: usize, values: &[Vec<i32>]) -> Self {
        Self {
            fa: HashMap::from_iter(values.iter().map(|x| ((x[0], x[1]), (x[0], x[1])))),
            cc: row * colum,
        }
    }
    fn find(&mut self, row_colum: (i32, i32)) -> (i32, i32) {
        if let Some(&target) = self.fa.get(&row_colum) {
            if target != row_colum {
                let x = self.find(target);
                self.fa.insert(row_colum, x);
            }
            return self.fa.get(&row_colum).copied().unwrap();
        }
        unreachable!()
    }
    fn union(&mut self, from: (i32, i32), to: (i32, i32)) -> bool {
        let a = self.find(from);
        let b = self.find(to);
        if a == b {
            return false;
        }
        self.fa.insert(a, b);
        self.cc -= 1;
        true
    }
}

struct DisjointSet2DFull {
    fa: Vec<Vec<(usize, usize)>>,
    cc: usize,
}
impl DisjointSet2DFull {
    fn new(row_col: (usize, usize)) -> Self {
        Self {
            fa: (0..row_col.0)
                .map(|x| (0..row_col.1).map(|y| (x, y)).collect())
                .collect(),
            cc: row_col.0 * row_col.1,
        }
    }
    fn find(&mut self, row_col: (usize, usize)) -> (usize, usize) {
        if self.fa[row_col.0][row_col.1] != row_col {
            self.fa[row_col.0][row_col.1] = self.find(self.fa[row_col.0][row_col.1]);
        }
        self.fa[row_col.0][row_col.1]
    }
    fn union(&mut self, from: (usize, usize), to: (usize, usize)) -> bool {
        let a = self.find(from);
        let b = self.find(to);
        if a == b {
            return false;
        }
        self.fa[a.0][a.1] = b;
        self.cc -= 1;
        true
    }
}
pub fn calc_equation(
    equations: Vec<Vec<String>>,
    values: Vec<f64>,
    queries: Vec<Vec<String>>,
) -> Vec<f64> {
    use std::collections::{HashMap, HashSet};
    let mut map: HashMap<String, f64> = HashMap::new();
    let mut chain: HashMap<String, HashSet<String>> = HashMap::new();
    let mut res = vec![];
    for i in 0..equations.len() {
        map.insert(format!("{}{}", equations[i][0], equations[i][1]), values[i]);
        map.insert(
            format!("{}{}", equations[i][1], equations[i][0]),
            1.0 / values[i],
        );
        chain
            .entry(equations[i][0].clone())
            .or_default()
            .insert(equations[i][1].clone());
        chain
            .entry(equations[i][1].clone())
            .or_default()
            .insert(equations[i][0].clone());
    }
    for q in queries {
        let mut selected = HashSet::new();
        if chain.contains_key(&q[0]) && chain.contains_key(&q[1]) {
            res.push(dfs_calc(1.0, &q[1], &q[0], &chain, &map, &mut selected));
        } else {
            res.push(-1.0);
        }
    }

    res
}
fn dfs_calc<'a>(
    currentvalue: f64,
    end: &str,
    current: &str,
    chain: &'a HashMap<String, HashSet<String>>,
    map: &HashMap<String, f64>,
    selected: &mut HashSet<&'a String>,
) -> f64 {
    if current == end {
        return currentvalue;
    }
    if let Some(next) = chain.get(current) {
        for n in next {
            if selected.contains(n) {
                continue;
            }
            selected.insert(n);
            let c_n = map.get(&format!("{}{}", current, n)).copied().unwrap() * currentvalue;
            let res = dfs_calc(c_n, end, n, chain, map, selected);
            if res != -1.0 {
                return res;
            }
            selected.remove(n);
        }
    }
    -1.0
}

pub fn maximum_segment_sum(nums: Vec<i32>, remove_queries: Vec<i32>) -> Vec<i64> {
    let mut current_nums = vec![0; nums.len()];
    let mut res = vec![0];
    let mut d_set = DisjointSet::new(remove_queries.len());
    for idx in remove_queries[1..].iter().map(|x| *x as usize).rev() {
        current_nums[idx] = nums[idx];
        d_set.value[idx] = nums[idx] as i64;
        let mut pre = res[res.len() - 1].max(nums[idx] as _);
        if idx > 0 && current_nums[idx - 1] != 0 {
            d_set.union(idx, idx - 1);
            pre = pre.max(d_set.get_value(idx));
        }
        if idx < current_nums.len() - 1 && current_nums[idx + 1] != 0 {
            d_set.union(idx, idx + 1);
            pre = pre.max(d_set.get_value(idx));
        }
        res.push(pre);
    }
    res.reverse();
    res
}

pub fn max_events(mut events: Vec<Vec<i32>>) -> i32 {
    let max_day = events.iter().map(|x| x[1]).max().unwrap();
    let mut d_set = DisjointSet::new(max_day as usize + 2);
    // events.sort_by(|a,b| a[0].cmp(&b[0]));
    events.sort_by_key(|x| x[1]);
    let mut res = 0;
    for e in events {
        if d_set.find(e[0] as usize) <= e[1] as usize {
            res += 1;
            let x = d_set.find(e[0] as usize);
            d_set.union(x, x + 1);
        }
    }
    res
}

pub fn avoid_flood(rains: Vec<i32>) -> Vec<i32> {
    use std::collections::{BTreeSet, HashMap};
    let mut res = vec![-1; rains.len()];
    let mut is_full: HashMap<i32, usize> = HashMap::new();
    let mut available_days: BTreeSet<usize> = rains
        .iter()
        .enumerate()
        .filter(|x| *x.1 == 0)
        .map(|x| x.0)
        .collect();
    for (i, r) in rains.into_iter().enumerate() {
        if r == 0 {
            continue;
        }
        if let Some(rain_day) = is_full.get_mut(&r) {
            if let Some(&ok_day) = available_days.range(*rain_day..i).next() {
                res[ok_day] = r;
                available_days.remove(&ok_day);
            } else {
                return vec![];
            }
        }
        is_full.insert(r, i);
    }
    for d in available_days {
        res[d] = 1;
    }
    res
}

pub fn largest_component_size(arr: Vec<i32>, m: i32) -> i32 {
    let m = m as usize;
    use std::collections::HashMap;
    let mut s = vec![0; arr.len() + 2];
    let mut d_set = DisjointSet::new(arr.len() + 2);
    let mut res = 1;
    let mut map = HashMap::new();
    for i in 0..arr.len() {
        let idx = arr[i] as usize;
        s[idx] = 1;
        *map.entry(1).or_default() += 1;
        println!("{:?}", s);
        if s[idx - 1] == 1 && s[idx + 1] == 1 {
            decrease_and_del_if_zero(&mut map, 1);
            decrease_and_del_if_zero(&mut map, d_set.get_size(idx + 1));
            decrease_and_del_if_zero(&mut map, d_set.get_size(idx - 1));
            d_set.union(idx - 1, idx);
            d_set.union(idx + 1, idx);
            *map.entry(d_set.get_size(idx)).or_default() += 1;
        } else if s[idx - 1] == 1 {
            decrease_and_del_if_zero(&mut map, 1);
            decrease_and_del_if_zero(&mut map, d_set.get_size(idx - 1));
            d_set.union(idx - 1, idx);
            *map.entry(d_set.get_size(idx)).or_default() += 1;
        } else if s[idx + 1] == 1 {
            decrease_and_del_if_zero(&mut map, 1);
            decrease_and_del_if_zero(&mut map, d_set.get_size(idx + 1));
            d_set.union(idx + 1, idx);
            *map.entry(d_set.get_size(idx)).or_default() += 1;
        }
        println!("{:?}", map);
        if map.contains_key(&m) {
            res = res.max(i + 1)
        }
    }
    res as _
}
fn decrease_and_del_if_zero(map: &mut HashMap<usize, usize>, key: usize) {
    if let Some(x) = map.get_mut(&key) {
        *x -= 1;
        if *x == 0 {
            map.remove(&key);
        }
    }
}

pub fn init_factors(MAXX: usize) -> Vec<Vec<usize>> {
    let mut fac = vec![vec![]; MAXX + 10];

    for i in 2..=MAXX {
        if fac[i].is_empty() {
            let mut j = i;
            while j <= MAXX {
                fac[j].push(i);
                j += i;
            }
        }
    }
    fac
}

pub fn can_traverse_all_pairs(nums: Vec<i32>) -> bool {
    let max = nums.iter().max().copied().unwrap();
    let fac = init_factors(max as usize);

    let mut d_set = DisjointSet::new(nums.len() + max as usize + 1);

    for i in 0..nums.len() {
        for &f in fac[nums[i] as usize].iter() {
            d_set.union(i, nums.len() + f);
        }
    }
    let mut set = std::collections::HashSet::new();
    for i in 0..nums.len() {
        set.insert(d_set.find(i));
    }
    set.len() == 1
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while a != 0 {
        let p = a;
        a = b % a;
        b = p
    }
    b
}

pub fn max_frequency(nums: Vec<i32>, k: i32, num_operations: i32) -> i32 {
    use std::collections::{BTreeMap, HashMap};
    let mut cnt = HashMap::new();
    let mut diff: BTreeMap<i32, i32> = BTreeMap::new();
    for n in nums {
        *cnt.entry(n).or_default() += 1;
        diff.entry(k).or_insert(0);
        *diff.entry(n - k).or_default() += 1;
        *diff.entry(n + k + 1).or_default() -= 1;
    }
    let mut sumD = 0;
    let mut ans = 0;
    for (k, v) in diff {
        sumD += v;
        ans = ans.max(sumD.min(cnt.get(&k).copied().unwrap_or(0) + num_operations));
    }
    ans
}

pub fn max_points(grid: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
    use std::collections::BinaryHeap;
    let mut queries: Vec<(usize, i32, i32)> = queries
        .into_iter()
        .enumerate()
        .map(|x| (x.0, x.1, 0))
        .collect();
    queries.sort_by(|a, b| a.1.cmp(&b.1));
    let mut heap = BinaryHeap::from([(-grid[0][0], 0_usize, 0_usize)]);
    let mut visited = vec![vec![false; grid[0].len()]; grid.len()];
    let mut current_score = 0;
    for i in 0..queries.len() {
        let max = queries[i].1;
        while let Some(&min) = heap.peek() {
            if -min.0 < max {
                heap.pop();
                visited[min.1][min.2] = true;
                current_score += 1;
                let adjacent_cells = [
                    (min.1 + 1, min.2),
                    (min.1 - 1, min.2),
                    (min.1, min.2 + 1),
                    (min.1, min.2 - 1),
                ];
                for (i, j) in adjacent_cells {
                    if i < grid.len() && j < grid[0].len() && !visited[i][j] {
                        heap.push((-grid[i][j], i, j));
                    }
                }
            } else {
                break;
            }
        }
        queries[i].2 = current_score;
    }
    queries.sort_by(|a, b| a.0.cmp(&b.0));
    queries.into_iter().map(|x| x.2).collect()
}

pub fn maximum_safeness_factor(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashSet;
    let mut current_distanse_to_theif = 0;
    let mut set = HashSet::new();
    let mut distanse_to_theif = vec![vec![usize::MAX; grid.len()]; grid.len()];
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            if grid[i][j] == 1 {
                set.insert((i, j));
            }
        }
    }
    let mut l = 0;
    let mut r = 0;
    while !set.is_empty() {
        let mut new_area = HashSet::new();
        r = r.max(current_distanse_to_theif);
        for &(i, j) in set.iter() {
            distanse_to_theif[i][j] = current_distanse_to_theif;
            let adjacent_areas = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
            for (adjacent_i, adjacent_j) in adjacent_areas {
                if adjacent_i < grid.len()
                    && adjacent_j < grid.len()
                    && distanse_to_theif[adjacent_i][adjacent_j] == usize::MAX
                    && !set.contains(&(adjacent_i, adjacent_j))
                {
                    new_area.insert((adjacent_i, adjacent_j));
                }
            }
        }
        set = new_area;
        current_distanse_to_theif += 1;
    }
    while l <= r {
        let mid = (r - l) / 2 + l;
        if check5(mid, &distanse_to_theif) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    l as i32
}
fn check5(distance: usize, distance_to_theif: &Vec<Vec<usize>>) -> bool {
    use std::collections::HashSet;
    let mut visited = HashSet::new();
    check6(distance, distance_to_theif, 0, 0, &mut visited)
}
fn check6(
    distance: usize,
    distance_to_theif: &Vec<Vec<usize>>,
    i: usize,
    j: usize,
    visited: &mut HashSet<(usize, usize)>,
) -> bool {
    if i == j && i == distance_to_theif.len() - 1 {
        return distance_to_theif[i][j] >= distance;
    }
    if visited.contains(&(i, j)) {
        return false;
    }
    if i >= distance_to_theif.len() || j >= distance_to_theif.len() {
        return false;
    }
    if distance_to_theif[i][j] < distance {
        return false;
    }
    visited.insert((i, j));
    let adjacent_cells = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
    for adjacent_cell in adjacent_cells {
        if check6(
            distance,
            distance_to_theif,
            adjacent_cell.0,
            adjacent_cell.1,
            visited,
        ) {
            return true;
        }
    }

    false
}

pub fn regions_by_slashes(mut grid: Vec<String>) -> i32 {
    let mut d_set = DisjointSet2DFull::new((grid.len() + 1, grid.len() + 1));
    for i in 0..grid.len() {
        d_set.union((0, i), (0, i + 1));
        d_set.union((grid.len(), i), (grid.len(), i + 1));
        d_set.union((i, grid.len()), (i + 1, grid.len()));
        d_set.union((i, 0), (i + 1, 0));
    }

    let mut res = 1;
    for (i, g) in grid.into_iter().enumerate() {
        for (j, c) in g.chars().enumerate() {
            if c == '/' {
                if !d_set.union((i + 1, j), (i, j + 1)) {
                    res += 1;
                }
            } else if c == '\\' && !d_set.union((i, j), (i + 1, j + 1)) {
                res += 1;
            }
        }
    }
    res
}

pub fn max_num_edges_to_remove(n: i32, edges: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    let mut d_set_alice = DisjointSet::new(n as usize);
    let mut d_set_bob = DisjointSet::new(n as usize);
    let mut edges_alice = vec![];
    let mut edges_bob = vec![];
    let mut edges_common = vec![];
    for edge in edges {
        if edge[0] == 1 {
            edges_alice.push(edge);
        } else if edge[0] == 2 {
            edges_bob.push(edge);
        } else {
            edges_common.push(edge);
        }
    }
    for e in edges_common {
        d_set_bob.union(e[1] as usize - 1, e[2] as usize - 1);
        if !d_set_alice.union(e[1] as usize - 1, e[2] as usize - 1) {
            res += 1;
        }
    }
    for e in edges_alice {
        if !d_set_alice.union(e[1] as usize - 1, e[2] as usize - 1) {
            res += 1;
        }
    }

    for e in edges_bob {
        if !d_set_bob.union(e[1] as usize - 1, e[2] as usize - 1) {
            res += 1;
        }
    }
    if d_set_alice.cc != 1 || d_set_bob.cc != 1 {
        return -1;
    }
    res as _
}

pub fn friend_requests(n: i32, restrictions: Vec<Vec<i32>>, requests: Vec<Vec<i32>>) -> Vec<bool> {
    use std::collections::{HashMap, HashSet};
    let mut restriction_map: HashMap<usize, HashSet<usize>> = HashMap::new();
    for r in restrictions {
        restriction_map
            .entry(r[1] as usize)
            .or_default()
            .insert(r[0] as usize);
        restriction_map
            .entry(r[0] as usize)
            .or_default()
            .insert(r[1] as usize);
    }
    let mut d_set = DisjointSet::new(n as usize);
    let mut res = vec![false; requests.len()];
    for (i, r) in requests.iter().enumerate() {
        let target_friends = d_set.find(r[1] as usize);
        let mut is_ok = true;
        if let Some(dislike) = restriction_map.get(&d_set.find(r[0] as usize)) {
            for i in 0..n as usize {
                if d_set.find(i) == target_friends && dislike.contains(&i) {
                    is_ok = false;
                    break;
                }
            }
        }
        let target_friends = d_set.find(r[0] as usize);
        if let Some(dislike) = restriction_map.get(&d_set.find(r[0] as usize)) {
            for i in 0..n as usize {
                if d_set.find(i) == target_friends && dislike.contains(&i) || !is_ok {
                    is_ok = false;
                    break;
                }
            }
        }
        if is_ok {
            let empty = HashSet::new();
            let a = restriction_map
                .get(&d_set.find(r[1] as usize))
                .unwrap_or(&empty);
            let b = restriction_map
                .get(&d_set.find(r[0] as usize))
                .unwrap_or(&empty);
            let res = a.union(b).copied().collect();
            d_set.union(r[1] as usize, r[0] as usize);
            let c = restriction_map.get_mut(&d_set.find(r[1] as usize)).unwrap();
            restriction_map.insert(d_set.find(r[1] as usize), res);
        }
        res[i] = is_ok;
    }
    res
}

pub fn latest_day_to_cross(row: i32, col: i32, mut cells: Vec<Vec<i32>>) -> i32 {
    let mut grid = vec![vec![1; col as usize]; row as usize];

    let mut d_set = DisjointSet2DFull::new((row as usize, col as usize));
    while let Some(cell) = cells.pop() {
        let (row, col) = (cell[0] - 1, cell[1] - 1);
        grid[row as usize][col as usize] = 0;
        let directions = [
            (row + 1, col),
            (row - 1, col),
            (row, col - 1),
            (row, col + 1),
        ];
        for d in directions {
            if d.0 >= 0
                && (d.0 as usize) < grid.len()
                && d.1 >= 0
                && (d.1 as usize) < grid[1].len()
                && grid[d.0 as usize][d.1 as usize] == 0
            {
                d_set.union((d.0 as usize, d.1 as usize), (row as usize, col as usize));
            }
        }
        use std::collections::HashSet;
        let mut is_ok = false;
        let mut set = HashSet::new();
        for i in 0..grid[0].len() {
            set.insert(d_set.find((grid.len() - 1, i)));
        }
        for i in 0..grid[0].len() {
            if set.contains(&d_set.find((0, i))) {
                is_ok = true;
                break;
            }
        }
        if is_ok {
            break;
        }
    }
    cells.len() as _
}

pub fn num_similar_groups(strs: Vec<String>) -> i32 {
    let mut d_set = DisjointSet::new(strs.len());
    for i in 0..strs.len() {
        for j in 0..strs.len() {
            if is_similar(strs[i].as_bytes(), strs[j].as_bytes()) {
                d_set.union(i, j);
            }
        }
    }
    d_set.cc as _
}
fn is_similar(s1: &[u8], s2: &[u8]) -> bool {
    let mut diff = 0;
    for i in 0..s1.len() {
        if s1[i] != s2[i] {
            diff += 1;
        }
        if diff > 2 {
            return false;
        }
    }
    true
}

pub fn remove_stones(stones: Vec<Vec<i32>>) -> i32 {
    let max_row = stones.iter().map(|x| x[0]).max().unwrap() as usize;
    let max_column = stones.iter().map(|x| x[1]).max().unwrap() as usize;

    let mut d_set_2d = DisjointSet2D::new(max_row, max_column, &stones);
    for i in 0..stones.len() {
        for j in 0..stones.len() {
            if stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1] {
                d_set_2d.union((stones[i][0], stones[i][1]), (stones[j][0], stones[j][1]));
            }
        }
    }
    ((stones.len() * stones[0].len()) - d_set_2d.cc) as i32
}

pub fn accounts_merge(accounts: Vec<Vec<String>>) -> Vec<Vec<String>> {
    use std::collections::HashSet;
    let mut d_set = DisjointSet::new(accounts.len());
    let mut accounts_sets: Vec<HashSet<&String>> = accounts
        .iter()
        .map(|x| HashSet::from_iter(x[1..].iter()))
        .collect();
    for i in 0..accounts_sets.len() - 1 {
        for j in i + 1..accounts_sets.len() {
            if accounts_sets[i].intersection(&accounts_sets[j]).count() > 0 {
                d_set.union(i, j);
            }
        }
    }
    for i in 0..accounts.len() {
        d_set.find(i);
    }
    println!("{:?}", d_set.fa);
    println!("{:?}", accounts_sets);
    for i in 0..accounts.len() {
        let fa = d_set.find(i);
        if fa != i {
            accounts_sets[fa] = (accounts_sets[fa].union(&accounts_sets[i]))
                .copied()
                .collect();
            accounts_sets[i].clear();
        }
    }
    println!("{:?}", accounts_sets);

    let mut res: Vec<Vec<String>> = accounts.iter().map(|x| vec![x[0].clone()]).collect();
    for i in 0..accounts.len() {
        if !accounts_sets[i].is_empty() {
            let mut a = accounts_sets[i]
                .iter()
                .map(|x| String::from(x.as_str()))
                .collect::<Vec<String>>();
            a.sort();
            res[i].extend(a);
        }
    }
    let mut res: Vec<Vec<String>> = res.into_iter().filter(|x| x.len() > 1).collect();
    for x in res.iter_mut() {
        x.sort();
    }
    res
}

pub fn path_existence_queries(
    n: i32,
    nums: Vec<i32>,
    max_diff: i32,
    queries: Vec<Vec<i32>>,
) -> Vec<bool> {
    let mut l = 0;
    let mut r = 0;
    let mut d_set = DisjointSet::new(nums.len());
    while r < nums.len() {
        while r < nums.len() && nums[r] - nums[l] <= max_diff {
            d_set.union(r, l);
        }
        l = r;
    }
    println!("{:?}", d_set.fa);
    queries
        .into_iter()
        .map(|q| d_set.find(q[0] as usize) == d_set.find(q[1] as usize))
        .collect()
}

pub fn min_swaps(mut nums: Vec<i32>) -> i32 {
    let mut d_set = DisjointSet::new(nums.len());
    let mut nums = nums
        .into_iter()
        .enumerate()
        .map(|(i, n)| (i, n, parse(n)))
        .collect::<Vec<_>>();
    nums.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)));
    for (i, (pre_i, pre_n, n)) in nums.into_iter().enumerate() {
        // if n != pre_n {
        d_set.union(i, pre_i);
        // }
    }
    let mut res = 0;
    for i in 0..d_set.size.len() {
        res += (d_set.get_size(i) - 1);
    }
    res as _
}
fn parse(mut num: i32) -> i32 {
    let mut res = 0;
    while num != 0 {
        res += num % 10;
        num /= 10;
    }
    res
}

pub fn minimum_operations(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    use std::collections::VecDeque;

    let mut q = VecDeque::from([root]);
    let mut res = 0;
    while !q.is_empty() {
        let mut p = q.split_off(0);
        let values: Vec<i32> = p
            .iter()
            .filter(|x| x.is_some())
            .map(|x| x.as_ref().unwrap().clone())
            .map(|x| x.borrow().val)
            .collect();
        res += calc(values);
        for n in p {
            if let Some(n) = n.as_ref() {
                q.push_back(n.borrow().left.clone());
                q.push_back(n.borrow().right.clone());
            }
        }
    }
    res
}
fn calc(values: Vec<i32>) -> i32 {
    let mut t = DisjointSet::new(values.len());
    let mut values: Vec<(usize, i32)> = values.into_iter().enumerate().collect();
    values.sort_by(|a, b| a.1.cmp(&b.1));
    for (i, (pre_i, n)) in values.into_iter().enumerate() {
        t.union(pre_i, i);
    }
    (t.size.len() - t.cc) as _
}

pub fn smallest_string_with_swaps(s: String, pairs: Vec<Vec<i32>>) -> String {
    use std::collections::HashMap;

    let mut d_set = DisjointSet::new(s.len());
    for p in pairs {
        d_set.union(p[0] as usize, p[1] as usize);
    }
    let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..s.len() {
        map.entry(d_set.find(i)).or_default().push(i);
    }
    let mut res = s.chars().map(|x| x.to_string()).collect::<Vec<_>>();
    for indexes in map.values_mut() {
        indexes.sort();
        let mut chars = vec![];
        for i in indexes.iter() {
            chars.push(res[*i].to_string());
        }
        chars.sort();
        for (char_i, res_i) in indexes.iter().enumerate() {
            res[*res_i] = chars[char_i].to_string();
        }
    }
    res.join("")
}

pub fn smallest_equivalent_string(s1: String, s2: String, mut base_str: String) -> String {
    use std::collections::HashMap;
    let mut d_set = DisjointSet::new(26);
    for (a, b) in s1
        .chars()
        .zip(s2.chars())
        .map(|(a, b)| (a as u8 as usize - 97, b as u8 as usize - 97))
    {
        d_set.union(a, b);
    }
    let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..26 {
        map.entry(d_set.find(i)).or_default().push(i);
    }
    let mut min_map = ['z'; 26];
    for ccc in map.values() {
        let min = (ccc.iter().min().copied().unwrap() as u8 + 97) as char;
        for &c in ccc {
            min_map[c] = min
        }
    }
    unsafe {
        let m = base_str.as_mut_vec();
        for i in 0..m.len() {
            let idx = m[i] as usize - 97;
            m[i] = min_map[idx] as u8;
        }
    }
    base_str
}

pub fn minimum_hamming_distance(
    mut source: Vec<i32>,
    target: Vec<i32>,
    allowed_swaps: Vec<Vec<i32>>,
) -> i32 {
    use std::collections::HashMap;
    let mut d_set = DisjointSet::new(source.len());
    for swap in allowed_swaps {
        d_set.union(swap[0] as _, swap[1] as _);
    }
    let mut map: HashMap<usize, Vec<usize>> = HashMap::new();

    for i in 0..source.len() {
        map.entry(d_set.find(i)).or_default().push(i);
    }
    for indexes in map.values() {
        let mut available: HashMap<i32, i32> = HashMap::new();
        for &idx in indexes {
            *available.entry(source[idx]).or_default() += 1;
        }
        for &idx in indexes {
            if source[idx] != target[idx] {
                if let Some(x) = available.get_mut(&target[idx]) {
                    source[idx] = target[idx];
                    *x -= 1;
                    if *x == 0 {
                        available.remove(&target[idx]);
                    }
                } else {
                    source[idx] = -1;
                }
            } else if let Some(x) = available.get_mut(&source[idx]) {
                *x -= 1;
                if *x == 0 {
                    available.remove(&source[idx]);
                }
            }
        }
    }
    source
        .into_iter()
        .zip(target)
        .filter(|(a, b)| a != b)
        .count() as _
}

pub fn min_time(n: i32, edges: Vec<Vec<i32>>, k: i32) -> i32 {
    let mut l = 0;
    let mut r = edges.iter().map(|x| x[1]).max().unwrap();
    while l <= r {
        let mid = (r - l) / 2 + l;
        if check(n as _, edges.as_ref(), k, mid) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    l
}
fn check(n: usize, edges: &Vec<Vec<i32>>, k: i32, time: i32) -> bool {
    let mut d_set = DisjointSet::new(n);
    for e in edges.iter().filter(|x| x[2] < time) {
        d_set.union(e[0] as _, e[1] as _);
    }
    d_set.cc >= k as _
}

pub fn min_cost(n: i32, edges: Vec<Vec<i32>>, k: i32) -> i32 {
    if edges.is_empty() {
        return 0;
    }
    let mut l = 0;
    let mut r = edges.iter().map(|x| x[2]).max().unwrap();
    while l <= r {
        let mid = (r - l) / 2 + l;
        if check2(n as _, edges.as_ref(), k, mid) {
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    l
}

fn check2(n: usize, edges: &Vec<Vec<i32>>, k: i32, mid: i32) -> bool {
    let mut d_set = DisjointSet::new(n);
    for e in edges.iter().filter(|x| x[2] <= mid) {
        d_set.union(e[0] as _, e[1] as _);
    }
    d_set.cc <= k as _
}
pub fn swim_in_water(grid: Vec<Vec<i32>>) -> i32 {
    let mut l = 0;
    let mut r = grid
        .iter()
        .map(|x| x.iter().max().unwrap())
        .max()
        .copied()
        .unwrap();
    while l <= r {
        let mid = (r - l) / 2 + l;
        println!("{}", mid);
        if check3(&grid, mid) {
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    l
}
fn check3(grid: &Vec<Vec<i32>>, mid: i32) -> bool {
    let mut visited = vec![vec![false; grid[0].len()]; grid.len()];
    dfs(grid, 0, 0, mid, &mut visited)
}
fn dfs(grid: &Vec<Vec<i32>>, i: i32, j: i32, mid: i32, visited: &mut Vec<Vec<bool>>) -> bool {
    if i < 0 || i as usize >= grid.len() || j < 0 || j as usize >= grid[0].len() {
        return false;
    }
    let mut i_usize = i as usize;
    let mut j_usize = j as usize;
    if grid[i_usize][j_usize] > mid || visited[i_usize][j_usize] {
        return false;
    }
    if i_usize == j_usize && i_usize == grid.len() - 1 {
        return true;
    }
    visited[i_usize][j_usize] = true;
    let dir = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
    for d in dir {
        if dfs(grid, d.0, d.1, mid, visited) {
            return true;
        }
    }
    false
}

pub fn max_alternating_sum(mut nums: Vec<i32>, swaps: Vec<Vec<i32>>) -> i64 {
    use std::collections::HashMap;
    let mut d_set = DisjointSet::new(nums.len());
    for swap in swaps {
        d_set.union(swap[0] as _, swap[1] as _);
    }
    let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..nums.len() {
        map.entry(d_set.find(i)).or_default().push(i);
    }
    for indexes in map.values_mut() {
        let mut available_nums = Vec::with_capacity(indexes.len());
        for i in indexes.iter() {
            available_nums.push(nums[*i]);
        }
        indexes.sort_by(|a, b| (b % 0).cmp(&(a % 0)));
        available_nums.sort();
        for i in (0..indexes.len()).rev() {
            nums[i] = available_nums[i];
        }
    }
    let mut res = 0;
    for i in 0..nums.len() {
        res += if i % 2 == 0 {
            nums[i] as i64
        } else {
            -nums[i] as i64
        }
    }
    res
}

pub fn min_swaps_couples(row: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut need_change = vec![];
    let mut value_idx = HashMap::new();
    for i in (1..row.len()).step_by(2) {
        if row[i - 1]
            != if row[i] % 2 == 0 {
                row[i] + 1
            } else {
                row[i] - 1
            }
        {
            need_change.push(row[i - 1]);
            value_idx.insert(row[i - 1], need_change.len() - 1);
            need_change.push(row[i]);
            value_idx.insert(row[i], need_change.len() - 1);
        }
    }
    let origin = value_idx.clone();
    for i in (1..need_change.len()).step_by(2) {
        let target = if need_change[i - 1] % 2 == 0 {
            need_change[i - 1] + 1
        } else {
            need_change[i - 1] - 1
        };
        let target_idx = value_idx.get(&target).copied().unwrap();
        need_change.swap(i, target_idx);
        value_idx.insert(target, i);
        value_idx.insert(need_change[i], target_idx);
    }
    let mut d_set = DisjointSet::new(need_change.len());
    for i in (1..need_change.len()).step_by(2) {
        let o_idx = origin.get(&need_change[i]).copied().unwrap();
        d_set.union(i, o_idx);
    }

    (d_set.fa.len() - d_set.cc) as _
}

pub fn find_redundant_directed_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
    let mut in_ = vec![0; edges.len()];
    for i in 0..edges.len() {
        in_[edges[i][1] as usize - 1] += 1;
    }

    for i in (0..edges.len()).rev() {
        if in_[edges[i][1] as usize] > 1 && remove_is_tree(&edges, i) {
            return edges[i].clone();
        }
    }
    getRemoveEdge(edges)
}
fn getRemoveEdge(edges: Vec<Vec<i32>>) -> Vec<i32> {
    let mut d_set = DisjointSet::new(edges.len());
    for e in edges {
        if !d_set.union(e[0] as usize - 1, e[1] as usize - 1) {
            return e;
        }
    }
    unreachable!()
}
fn remove_is_tree(edges: &Vec<Vec<i32>>, remove_idx: usize) -> bool {
    let mut d_set = DisjointSet::new(edges.len());
    for (i, e) in edges.iter().enumerate() {
        if i == remove_idx {
            continue;
        }
        d_set.union(e[0] as usize - 1, e[1] as usize - 1);
    }
    let mut root = d_set.find(0);
    for i in 0..edges.len() {
        if d_set.find(i) != root {
            return false;
        }
    }
    true
}
