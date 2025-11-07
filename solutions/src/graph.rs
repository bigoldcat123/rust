use std::{
    char::ToUppercase,
    collections::{HashMap, HashSet},
};

struct DSet {
    fa: Vec<usize>,
    size: Vec<usize>,
    cc: usize,
}
impl DSet {
    fn new(len: usize) -> Self {
        Self {
            fa: (0..len).collect(),
            cc: len,
            size: vec![1; len],
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.fa[x] != x {
            self.fa[x] = self.find(self.fa[x]);
        }
        self.fa[x]
    }
    /// true  => union
    ///
    /// flase => alread unioned!
    fn union(&mut self, from: usize, to: usize) -> bool {
        let a = self.find(from);
        let b = self.find(to);
        if a == b {
            return false;
        }
        self.fa[a] = b;
        self.size[b] += self.size[a];
        self.cc -= 1;
        true
    }
    fn get_size(&mut self, x: usize) -> usize {
        let x = self.find(x);
        self.size[x]
    }
}
fn add(b:i32,a:i32) -> i32 {
    let a = a + b;
    a
}

pub fn color_border(mut grid: Vec<Vec<i32>>, row: i32, col: i32, color: i32) -> Vec<Vec<i32>> {
    let orign_color = grid[row as usize][col as usize];
    let mut selected = vec![vec![false; grid[0].len()]; grid.len()];
    for i in 0..grid.len() {
        for j in 0..grid.len() {
            if grid[i][j] == orign_color {
                dfs_color_border(&mut grid, i as i32, j as i32, color, &mut selected);
            }
        }
    }
    grid
}
fn dfs_color_border(
    grid: &mut Vec<Vec<i32>>,
    i: i32,
    j: i32,
    color: i32,
    selected: &mut Vec<Vec<bool>>,
) {
    if selected[i as usize][j as usize] {
        return;
    }
    selected[i as usize][j as usize] = true;
    let des = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
    for (next_i, next_j) in des {
        if next_i < 0
            || next_j < 0
            || (next_i as usize) >= grid.len()
            || (next_j as usize) >= grid[0].len()
        {
            grid[i as usize][i as usize] = color;
        } else {
            if grid[next_i as usize][next_j as usize] != grid[i as usize][j as usize] {
                grid[i as usize][j as usize] = color;
            } else {
                dfs_color_border(grid, next_i, next_j, color, selected);
            }
        }
    }
}

pub fn flood_fill(mut image: Vec<Vec<i32>>, sr: i32, sc: i32, color: i32) -> Vec<Vec<i32>> {
    let origin_color = image[sr as usize][sc as usize];
    if origin_color != color {
        dfs_flood_fill(&mut image, sr, sc, origin_color, color);
    }
    image
}
fn dfs_flood_fill(image: &mut Vec<Vec<i32>>, i: i32, j: i32, orign_color: i32, target_color: i32) {
    if i >= 0 && j >= 0 && (i as usize) < image.len() && (j as usize) < image[0].len() {
        let i_usize = i as usize;
        let j_usize = j as usize;
        if image[i_usize][j_usize] == orign_color {
            image[i_usize][j_usize] = target_color;
            let des = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
            for (i, j) in des {
                dfs_flood_fill(image, i, j, orign_color, target_color);
            }
        }
    }
}

pub fn find_max_fish(grid: Vec<Vec<i32>>) -> i32 {
    let mut ans = 0;
    let mut selected = grid
        .iter()
        .map(|x| x.iter().map(|_| false).collect())
        .collect();
    for i in 0..grid.len() {
        for j in 0..grid[i].len() {
            ans = ans.max(dfs_find_max_fish(i as i32, j as i32, &grid, &mut selected))
        }
    }
    ans
}
fn dfs_find_max_fish(i: i32, j: i32, grid: &Vec<Vec<i32>>, selected: &mut Vec<Vec<bool>>) -> i32 {
    if i >= 0 && j >= 0 && (i as usize) < grid.len() && (j as usize) < grid[0].len() {
        let mut i_usize = i as usize;
        let mut j_usize = j as usize;
        if selected[i_usize][j_usize] {
            return 0;
        }
        selected[i_usize][j_usize] = true;
        if grid[i_usize][j_usize] == 0 {
            return 0;
        } else {
            let mut ans = grid[i_usize][j_usize];
            let des = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
            for (i, j) in des {
                ans += dfs_find_max_fish(i, j, grid, selected)
            }
            return ans;
        }
    } else {
        0
    }
}

pub fn island_perimeter(grid: Vec<Vec<i32>>) -> i32 {
    let mut ans = 0;
    for i in 0..grid.len() {
        for j in 0..grid.len() {
            if grid[i][j] == 1 {
                if i == 0 {
                    ans += 1;
                }
                if i == grid.len() - 1 {
                    ans += 1
                }
                if j == 0 {
                    ans += 1;
                }
                if j == grid[0].len() - 1 {
                    ans += 1;
                }
            } else {
                let i = i as i32;
                let j = j as i32;
                let des = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
                for (i, j) in des {
                    if i >= 0 && (i as usize) < grid.len() && j >= 0 && (j as usize) < grid[0].len()
                    {
                        if grid[i as usize][j as usize] == 1 {
                            ans += 1;
                        }
                    }
                }
            }
        }
    }
    ans
}

pub fn largest_area(grid: Vec<String>) -> i32 {
    let mut ans = 0;
    let grid: Vec<&[u8]> = grid.iter().map(|x| x.as_bytes()).collect();
    for i in 0..grid.len() {
        for j in 0..grid[i].len() {
            if grid[i][j] != b'0' {
                let (area, not_aligned) = dfs_largest_area(
                    i as i32,
                    j as i32,
                    1,
                    &grid,
                    &mut vec![vec![false; grid[0].len()]; grid.len()],
                );
                if not_aligned {
                    ans = ans.max(area)
                }
            }
        }
    }
    ans
}
fn dfs_largest_area(
    i: i32,
    j: i32,
    area_id: u8,
    grid: &Vec<&[u8]>,
    visited: &mut Vec<Vec<bool>>,
) -> (i32, bool) {
    if i >= 0 && (i as usize) < grid.len() && j > 0 && (j as usize) < grid[0].len() {
        let i_usize = i as usize;
        let j_usize = j as usize;
        if visited[i_usize][j_usize] {
            return (0, true);
        }
        visited[i_usize][j_usize] = true;
        if grid[i_usize][j_usize] == area_id {
            let mut ans = 1;
            let mut not_aligned = true;
            let des = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
            for (i, j) in des {
                let (a, b) = dfs_largest_area(i, j, area_id, grid, visited);
                ans += a;
                not_aligned &= b;
            }
            (ans, not_aligned)
        } else if grid[i_usize][j_usize] == b'0' {
            (0, false)
        } else {
            (0, true)
        }
    } else {
        (0, false)
    }
}

pub fn num_islands(grid: Vec<Vec<char>>) -> i32 {
    use std::collections::HashSet;
    let mut res = 0;
    let mut selected = HashSet::new();
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            if dfs_num_islands(i as i32, j as i32, &mut selected, &grid) {
                res += 1;
            }
        }
    }
    res
}
fn dfs_num_islands(
    i: i32,
    j: i32,
    selected: &mut HashSet<(i32, i32)>,
    grid: &Vec<Vec<char>>,
) -> bool {
    if (i >= 0 && (i as usize) < grid.len() && j >= 0 && (j as usize) < grid[0].len()) {
        if grid[i as usize][j as usize] == '0' {
            return false;
        }
        if selected.insert((i, j)) {
            let next = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
            for &(x, y) in next.iter() {
                dfs_num_islands(x, y, selected, grid);
            }
            true
        } else {
            false
        }
    } else {
        false
    }
}

pub fn max_candies(
    status: Vec<i32>,
    candies: Vec<i32>,
    keys: Vec<Vec<i32>>,
    contained_boxes: Vec<Vec<i32>>,
    initial_boxes: Vec<i32>,
) -> i32 {
    let mut obtained_boxs = HashSet::from_iter(initial_boxes.into_iter().map(|x| x as usize));
    let mut obtained_keys = HashSet::new();
    let mut res = 0;
    dfs_max_candies(
        obtained_boxs,
        obtained_keys,
        &status,
        &candies,
        &keys,
        &contained_boxes,
        &mut res,
    );
    res
}
fn dfs_max_candies(
    obtained_boxs: HashSet<usize>,
    mut obtained_keys: HashSet<usize>,
    status: &Vec<i32>,
    candies: &Vec<i32>,
    keys: &Vec<Vec<i32>>,
    contained_boxes: &Vec<Vec<i32>>,
    res: &mut i32,
) {
    let mut next_boxs = HashSet::new();
    let mut new_thing = false;
    for b in obtained_boxs {
        if status[b] == 1 {
            *res += candies[b];
            for &k in keys[b].iter() {
                obtained_keys.insert(k as usize);
            }
            for b in contained_boxes[b].iter() {
                next_boxs.insert(*b as usize);
            }
            new_thing = true;
        } else {
            if obtained_keys.contains(&b) {
                *res += candies[b];
                for &k in keys[b].iter() {
                    obtained_keys.insert(k as usize);
                }
                for b in contained_boxes[b].iter() {
                    next_boxs.insert(*b as usize);
                }
                new_thing = true;
            } else {
                next_boxs.insert(b);
            }
        }
    }
    if new_thing {
        dfs_max_candies(
            next_boxs,
            obtained_keys,
            status,
            candies,
            keys,
            contained_boxes,
            res,
        );
    }
}

pub fn num_ways(n: i32, relation: Vec<Vec<i32>>, k: i32) -> i32 {
    use std::collections::HashMap;
    let mut map: HashMap<i32, Vec<i32>> = HashMap::new();

    for r in relation {
        map.entry(r[0]).or_default().push(r[1]);
    }
    println!("{:?}", map);
    let mut res = 0;

    dfs_num_ways(0, k, &map, &mut res, 0, n - 1);

    res
}
fn dfs_num_ways(
    current_turn: i32,
    max_turn: i32,
    relation: &HashMap<i32, Vec<i32>>,
    res: &mut i32,
    current_people: i32,
    last: i32,
) {
    println!("{} {}", current_people, last);
    if current_people == last {
        *res += 1
    }
    if current_turn > max_turn {
        return;
    }
    if let Some(next_people) = relation.get(&current_people) {
        for &n in next_people {
            dfs_num_ways(current_turn + 1, max_turn, relation, res, n, last);
        }
    }
}

pub fn minimum_cost(n: i32, edges: Vec<Vec<i32>>, query: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::HashMap;
    let mut d_set = DSet::new(n as usize);
    for e in edges.iter() {
        d_set.union(e[0] as usize, e[1] as usize);
    }
    let mut map = HashMap::new();
    for i in 0..n as usize {
        map.insert(d_set.find(i), i32::MAX);
    }
    for e in edges {
        *map.entry(d_set.find(e[0] as usize)).or_default() &= e[2];
    }
    query
        .into_iter()
        .map(|x| {
            if d_set.find(x[0] as usize) == d_set.find(x[1] as usize) {
                let a = map[&d_set.find(x[0] as usize)];
                a
            } else {
                -1
            }
        })
        .collect()
}

pub fn process_queries(c: i32, connections: Vec<Vec<i32>>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::{BTreeSet, HashMap};
    let mut d_set = DSet::new(c as usize + 1);
    let mut is_offline = vec![false; c as usize + 1];
    for c in connections {
        d_set.union(c[0] as usize, c[1] as usize);
    }
    let mut grid: HashMap<usize, BTreeSet<i32>> = HashMap::new();
    for i in 1..=c {
        grid.entry(d_set.find(i as usize)).or_default().insert(i);
    }
    let mut res = vec![];
    for q in queries {
        let station = q[1];
        if q[0] == 1 {
            if is_offline[station as usize] {
                if let Some(grid_stations) = grid.get(&d_set.find(station as usize)) {
                    if let Some(s) = grid_stations.first() {
                        res.push(*s);
                    } else {
                        res.push(-1);
                    }
                } else {
                    res.push(-1);
                }
            } else {
                res.push(station);
            }
        } else {
            is_offline[station as usize] = true;
            if let Some(grid_stations) = grid.get_mut(&d_set.find(station as usize)) {
                grid_stations.remove(&station);
            }
        }
    }
    res
}

pub fn find_all_people(n: i32, meetings: Vec<Vec<i32>>, first_person: i32) -> Vec<i32> {
    use std::collections::{BTreeMap, HashSet};

    let mut d_set = DSet::new(n as usize);
    d_set.union(0, first_person as usize);
    let mut time_meeting_map: BTreeMap<i32, HashMap<i32, Vec<i32>>> = BTreeMap::new();
    for m in meetings {
        time_meeting_map
            .entry(m[2])
            .or_default()
            .entry(m[1])
            .or_default()
            .push(m[0]);
    }
    let mut res = HashSet::from([0, first_person]);
    for (_, m) in time_meeting_map {
        let mut calculated: HashSet<i32> = HashSet::new();
        for (&p, _) in m.iter() {
            let mut people = HashSet::new();
            if dfs_find_all_people(&m, &mut d_set, &mut calculated, &mut people, p) {
                for p in people {
                    d_set.union(p as usize, 0);
                    res.insert(p);
                }
            }
        }
    }
    res.into_iter().collect()
}

fn dfs_find_all_people(
    meeting: &HashMap<i32, Vec<i32>>,
    dset: &mut DSet,
    selected: &mut HashSet<i32>,
    meeting_prople: &mut HashSet<i32>,
    current_people: i32,
) -> bool {
    if selected.insert(current_people) {
        meeting_prople.insert(current_people);
        let mut r = dset.find(current_people as usize) == dset.find(0);
        if let Some(x) = meeting.get(&current_people) {
            for meeting_with in x {
                r |= dfs_find_all_people(meeting, dset, selected, meeting_prople, *meeting_with);
            }
        }

        r
    } else {
        dset.find(current_people as usize) == dset.find(0)
    }
}

pub fn min_malware_spread2(graph: Vec<Vec<i32>>, initial: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let mut res = 0;
    let mut min = usize::MAX;
    for &remove_idx in initial.iter() {
        let mut d_set = DSet::new(graph.len());
        for i in 0..graph.len() - 1 {
            if i == remove_idx as usize {
                continue;
            }
            for j in i + 1..graph.len() {
                if j == remove_idx as usize {
                    continue;
                }
                if graph[i][j] == 1 {
                    d_set.union(i, j);
                }
            }
        }
        let mut effected = 0;
        let mut added = HashSet::new();
        for &i in initial.iter() {
            if i != remove_idx {
                if added.insert(d_set.find(i as usize)) {
                    effected += d_set.get_size(i as usize);
                }
            }
        }
        if effected < min {
            min = effected;
            res = remove_idx;
        }
    }
    res
}

pub fn max_alternating_sum(nums: Vec<i32>, swaps: Vec<Vec<i32>>) -> i64 {
    use std::collections::{BTreeMap, HashMap};
    let mut d_set = DSet::new(nums.len());
    for s in swaps {
        d_set.union(s[0] as usize, s[1] as usize);
    }
    let mut m: HashMap<usize, BTreeMap<i32, i32>> = HashMap::new();
    for i in 0..nums.len() {
        *m.entry(d_set.find(i))
            .or_default()
            .entry(nums[i])
            .or_default() += 1;
    }
    let mut res = 0;
    for i in 0..nums.len() {
        if i % 2 == 0 {
            res += *m
                .entry(d_set.fa[i])
                .or_default()
                .last_entry()
                .unwrap()
                .key() as i64;
            *m.entry(d_set.fa[i])
                .or_default()
                .last_entry()
                .unwrap()
                .get_mut() -= 1;
            if *m
                .entry(d_set.fa[i])
                .or_default()
                .last_entry()
                .unwrap()
                .get()
                == 0
            {
                m.entry(d_set.fa[i]).or_default().pop_last();
            }
        } else {
            res -= *m
                .entry(d_set.fa[i])
                .or_default()
                .first_entry()
                .unwrap()
                .key() as i64;
            *m.entry(d_set.fa[i])
                .or_default()
                .first_entry()
                .unwrap()
                .get_mut() -= 1;
            if *m
                .entry(d_set.fa[i])
                .or_default()
                .first_entry()
                .unwrap()
                .get()
                == 0
            {
                m.entry(d_set.fa[i]).or_default().pop_first();
            }
        }
    }

    res
}

pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::HashSet;
    let mut safe_nodes = HashSet::new();
    let mut danger_nodes = HashSet::new();
    for i in 0..graph.len() {
        let mut selected = HashSet::new();
        dfs_eventual_safe_nodes(&graph, i, &mut safe_nodes, &mut danger_nodes, &mut selected);
    }
    let mut res: Vec<i32> = safe_nodes.into_iter().map(|x| x as i32).collect();
    res.sort();
    res
}
fn dfs_eventual_safe_nodes(
    graph: &Vec<Vec<i32>>,
    current_node: usize,
    safe_nodes: &mut HashSet<usize>,
    danger_nodes: &mut HashSet<usize>,
    selected: &mut HashSet<usize>,
) -> bool {
    if selected.insert(current_node) {
        if danger_nodes.contains(&current_node) {
            return false;
        }
        if safe_nodes.contains(&current_node) {
            return true;
        }
        for &n in graph[current_node].iter() {
            if !dfs_eventual_safe_nodes(graph, n as usize, safe_nodes, danger_nodes, selected) {
                danger_nodes.insert(current_node);
                return false;
            }
        }
        safe_nodes.insert(current_node);
        true
    } else {
        return false;
    }
}

pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
    use std::collections::{HashMap, HashSet};
    let mut pre: HashMap<i32, HashSet<i32>> = HashMap::new();
    for p in prerequisites {
        pre.entry(p[0]).or_default().insert(p[1]);
    }
    let mut cache = HashSet::new();
    for i in 0..num_courses {
        let mut selected = HashSet::new();
        if !dfs_can_finish(i, &pre, &mut cache, &mut selected) {
            return false;
        }
    }
    true
}
fn dfs_can_finish(
    current_course: i32,
    pre: &HashMap<i32, HashSet<i32>>,
    cache: &mut HashSet<i32>,
    selected: &mut HashSet<i32>,
) -> bool {
    if cache.contains(&current_course) {
        return true;
    }
    if selected.insert(current_course) {
        if let Some(req) = pre.get(&current_course) {
            for r in req {
                if !dfs_can_finish(*r, pre, cache, selected) {
                    return false;
                }
            }
        }
        cache.insert(current_course);
        true
    } else {
        false
    }
}
pub fn find_circle_num(is_connected: Vec<Vec<i32>>) -> i32 {
    let mut selected = vec![false; is_connected.len()];
    let mut res = 0;
    for i in 0..is_connected.len() {
        if !selected[i] {
            dfs_connect(i, &is_connected, &mut selected);
            res += 1
        }
    }
    res
}
fn dfs_connect(from: usize, is_connected: &Vec<Vec<i32>>, selected: &mut Vec<bool>) {
    selected[from] = true;
    for i in 0..is_connected.len() {
        if !selected[i] && is_connected[from][i] == 1 {
            dfs_connect(i, is_connected, selected);
        }
    }
}

pub fn valid_path(n: i32, edges: Vec<Vec<i32>>, source: i32, destination: i32) -> bool {
    let mut d_set = DSet::new(n as usize);
    for e in edges {
        d_set.union(e[0] as usize, e[1] as usize);
    }
    d_set.find(source as usize) == d_set.find(destination as usize)
}

pub fn all_paths_source_target(graph: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut res = vec![];
    let mut current = vec![0];
    dfs(&mut current, 0, &mut res, &graph);
    res
}
fn dfs(current: &mut Vec<i32>, currentnode: usize, res: &mut Vec<Vec<i32>>, graph: &Vec<Vec<i32>>) {
    if currentnode == graph.len() - 1 {
        res.push(current.clone());
    }
    for &n in graph[currentnode].iter() {
        current.push(n);
        dfs(current, n as usize, res, graph);
        current.pop();
    }
}

pub fn can_reach(arr: Vec<i32>, start: i32) -> bool {
    let mut selected = vec![false; arr.len()];
    let mut d_set = DSet::new(arr.len());
    dfs_can_reach(&mut d_set, start as usize, &mut selected, &arr);
    let mut zero_idx = vec![];
    for i in 0..arr.len() {
        if arr[i] == 0 {
            zero_idx.push(i);
        }
    }
    zero_idx
        .into_iter()
        .any(|x| d_set.find(start as usize) == d_set.find(x))
}
fn dfs_can_reach(s_set: &mut DSet, current: usize, selected: &mut Vec<bool>, arr: &Vec<i32>) {
    if !selected[current] {
        selected[current] = true;

        let diff = arr[current] as usize;
        if current >= diff {
            s_set.union(current, current - diff);
            dfs_can_reach(s_set, current - diff, selected, arr);
        }
        if current + diff < selected.len() {
            s_set.union(current, current + diff);
            dfs_can_reach(s_set, current + diff, selected, arr);
        }
    }
}

pub fn can_visit_all_rooms(rooms: Vec<Vec<i32>>) -> bool {
    let mut d_set = DSet::new(rooms.len());
    let mut visited = vec![false; rooms.len()];
    dfs_can_visit_all_rooms(0, &rooms, &mut d_set, &mut visited);
    d_set.cc == 1
}
fn dfs_can_visit_all_rooms(
    roomidx: usize,
    rooms: &Vec<Vec<i32>>,
    d_set: &mut DSet,
    visited: &mut Vec<bool>,
) {
    if visited[roomidx] {
        return;
    }
    visited[roomidx] = true;
    for &k in rooms[roomidx].iter() {
        d_set.union(roomidx, k as usize);
        dfs_can_visit_all_rooms(k as usize, rooms, d_set, visited);
    }
}
pub fn count_pairs(n: i32, edges: Vec<Vec<i32>>) -> i64 {
    use std::collections::HashMap;
    let mut d_set = DSet::new(n as usize);
    let mut res = n as usize * (n - 1) as usize / 2;
    for e in edges {
        d_set.union(e[0] as usize, e[1] as usize);
    }
    let mut map: HashMap<usize, usize> = HashMap::new();
    for i in 0..n as usize {
        map.insert(d_set.find(i), d_set.get_size(i));
    }
    for (_, size) in map {
        res -= size * (size - 1) / 2
    }
    res as _
}

pub fn make_connected(n: i32, connections: Vec<Vec<i32>>) -> i32 {
    let mut extra = 0;
    let mut d_set = DSet::new(n as usize);
    for c in connections {
        if !d_set.union(c[0] as usize, c[1] as usize) {
            extra += 1;
        }
    }
    let required_line = d_set.cc - 1;
    if required_line <= extra {
        required_line as _
    } else {
        -1
    }
}

pub fn min_score(n: i32, roads: Vec<Vec<i32>>) -> i32 {
    struct Dset {
        fa: Vec<usize>,
        min: Vec<i32>,
    }
    impl Dset {
        fn new(len: usize) -> Self {
            Self {
                fa: (0..len).collect(),
                min: vec![i32::MAX; len],
            }
        }
        fn find(&mut self, x: usize) -> usize {
            if self.fa[x] != x {
                self.fa[x] = self.find(self.fa[x]);
            }
            self.fa[x]
        }
        fn union(&mut self, from: usize, to: usize, distance: i32) -> bool {
            let a = self.find(from);
            let b = self.find(to);
            if a == b {
                return true;
            }
            self.fa[a] = b;
            self.min[b] = self.min[a].min(self.min[b]);
            self.min[b] = self.min[b].min(distance);
            true
        }
        fn min_dist(&mut self, x: usize) -> i32 {
            let idx = self.find(x);
            self.min[idx] as _
        }
    }
    let mut dset = Dset::new(n as usize + 1);
    for r in roads {
        dset.union(r[0] as usize, r[1] as usize, r[2]);
    }
    dset.min_dist(n as usize)
}

pub fn remaining_methods(n: i32, k: i32, invocations: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::{HashMap, HashSet};
    let mut selected = vec![false; n as usize];
    let mut invock: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut d_set = DSet::new(n as usize);

    for i in invocations {
        invock.entry(i[0]).or_default().push(i[1]);
        d_set.union(i[0] as usize, i[1] as usize);
    }

    let mut suspicious = HashSet::from([k]);

    dfs_mark_suspicious(&mut suspicious, k, &invock, &mut selected);
    let danger = d_set.find(k as usize);
    let mut res = (0..n).collect();
    for i in 0..n as usize {
        if !suspicious.contains(&(i as i32)) && d_set.find(i) == danger {
            return res;
        }
    }

    res.into_iter()
        .filter(|x| !suspicious.contains(x))
        .collect()
}
fn dfs_mark_suspicious(
    suspicious: &mut HashSet<i32>,
    current_invockor: i32,
    invock: &HashMap<i32, Vec<i32>>,
    selected: &mut Vec<bool>,
) {
    if !selected[current_invockor as usize] {
        selected[current_invockor as usize] = true;
        if let Some(next_methodes) = invock.get(&current_invockor) {
            for &m in next_methodes {
                suspicious.insert(m);
                dfs_mark_suspicious(suspicious, m, invock, selected);
            }
        }
    }
}

pub fn count_complete_components(n: i32, edges: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashMap, HashSet};
    let mut d_set = DSet::new(n as usize);
    let mut map: HashMap<i32, i32> = HashMap::new();
    for e in edges {
        d_set.union(e[0] as usize, e[1] as usize);
        *map.entry(e[0]).or_default() += 1;
        *map.entry(e[1]).or_default() += 1;
    }
    let mut ans = d_set.cc;
    let mut is_notOk: HashSet<usize> = HashSet::new();
    for i in 0..n as usize {
        let size = d_set.get_size(i);
        if *map.entry(i as i32).or_default() != size as i32 - 1 {
            is_notOk.insert(d_set.find(i));
        }
    }
    (ans - is_notOk.len()) as _
}

pub fn get_ancestors(n: i32, edges: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    use std::collections::HashMap;
    let mut p_v_map: HashMap<i32, Vec<i32>> = HashMap::new();
    for e in edges {
        p_v_map.entry(e[0]).or_default().push(e[1]);
    }
    let mut res = vec![vec![]; n as usize];
    for i in 0..n as usize {
        let mut visited = vec![false; n as usize];
        dfs_set_ancestors(&mut res, &p_v_map, i, i, &mut visited);
    }
    res
}
fn dfs_set_ancestors(
    res: &mut Vec<Vec<i32>>,
    p_v_map: &HashMap<i32, Vec<i32>>,
    p: usize,
    currentp: usize,
    selected: &mut Vec<bool>,
) {
    if !selected[currentp] {
        selected[currentp] = true;
        if let Some(children) = p_v_map.get(&(currentp as i32)) {
            for &c in children {
                if !selected[c as usize] {
                    res[c as usize].push(p as i32);
                    dfs_set_ancestors(res, p_v_map, p, c as usize, selected);
                }
            }
        }
    }
}

pub fn max_amount(
    initial_currency: String,
    pairs1: Vec<Vec<String>>,
    rates1: Vec<f64>,
    pairs2: Vec<Vec<String>>,
    rates2: Vec<f64>,
) -> f64 {
    let mut map1: HashMap<String, Vec<(String, f64)>> = HashMap::new();
    let mut map2: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    for (p, r) in pairs1.into_iter().zip(rates1) {
        map1.entry(p[0].clone())
            .or_default()
            .push((p[1].clone(), r));
        map1.entry(p[1].clone())
            .or_default()
            .push((p[0].clone(), 1.0 / r));
    }
    for (p, r) in pairs2.into_iter().zip(rates2) {
        map2.entry(p[0].clone())
            .or_default()
            .push((p[1].clone(), r));
        map2.entry(p[1].clone())
            .or_default()
            .push((p[0].clone(), 1.0 / r));
    }
    let mut res = 1.0;
    let mut selected: HashSet<String> = HashSet::new();
    dfs_first_day(
        &initial_currency,
        &initial_currency,
        &map1,
        &map2,
        1.0,
        &mut res,
        &mut selected,
    );
    res
}

fn dfs_first_day(
    init: &String,
    current_currency: &String,
    map1: &HashMap<String, Vec<(String, f64)>>,
    map2: &HashMap<String, Vec<(String, f64)>>,
    current_money: f64,
    res: &mut f64,
    selected: &mut HashSet<String>,
) {
    // *res = (*res).max(current_money);

    if selected.insert(current_currency.clone()) {
        // println!("{} {} ",current_currency,current_money);

        if let Some(next) = map1.get(current_currency) {
            for (c, r) in next {
                let mut selected2: HashSet<String> = HashSet::new();
                dfs_sec_day(init, c, map2, current_money * r, res, &mut selected2);
                dfs_first_day(init, c, map1, map2, current_money * r, res, selected);
            }
        }
        let mut selected2: HashSet<String> = HashSet::new();
        dfs_sec_day(
            init,
            current_currency,
            map2,
            current_money,
            res,
            &mut selected2,
        );
    }
}
fn dfs_sec_day(
    init: &String,
    current_currency: &String,
    map2: &HashMap<String, Vec<(String, f64)>>,
    current_money: f64,
    res: &mut f64,
    selected: &mut HashSet<String>,
) {
    if init == current_currency {
        *res = (*res).max(current_money);
    }
    if selected.insert(current_currency.clone()) {
        if let Some(next) = map2.get(current_currency) {
            for (c, r) in next {
                dfs_sec_day(init, c, map2, current_money * r, res, selected);
            }
        }
    }
}

pub fn min_malware_spread(graph: Vec<Vec<i32>>, initial: Vec<i32>) -> i32 {
    use std::collections::HashSet;
    let mut d_set = DSet::new(graph.len());
    for i in 0..graph.len() {
        for j in 0..graph.len() {
            if graph[i][j] == 1 {
                d_set.union(i, j);
            }
        }
    }
    let mut min_effect = usize::MAX;
    let mut ans = 0;
    for &i in initial.iter() {
        let mut effected = 0;
        let mut calculated = HashSet::new();
        for &j in initial.iter().filter(|&&x| x != i) {
            if calculated.insert(d_set.find(j as usize)) {
                effected += d_set.get_size(j as usize);
            }
        }
        if effected < min_effect {
            min_effect = effected;
            ans = i;
        }
    }
    ans
}

pub fn maximum_detonation(bombs: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashMap, HashSet};
    let mut cache = HashMap::new();
    let mut res = 0;
    for b in 0..bombs.len() {
        let mut effected = HashSet::from([b]);
        dfs_maximum_detonation(b, &bombs, &cache, &mut effected);
        res = res.max(effected.len());
        cache.insert(b, effected);
    }
    res as _
}
fn dfs_maximum_detonation(
    current_bomb: usize,
    bombs: &Vec<Vec<i32>>,
    cache: &HashMap<usize, HashSet<usize>>,
    effected: &mut HashSet<usize>,
) {
    for (i, b) in bombs.iter().enumerate() {
        if (bombs[current_bomb][0] as i64 - b[0] as i64).pow(2)
            + ((bombs[current_bomb][1] - b[1]) as i64).pow(2)
            < (bombs[current_bomb][2] as i64).pow(2)
            && effected.insert(i)
        {
            if let Some(x) = cache.get(&i) {
                for &z in x {
                    effected.insert(z);
                }
            } else {
                dfs_maximum_detonation(i, bombs, cache, effected);
            }
        }
    }
}

pub fn accounts_merge(accounts: Vec<Vec<String>>) -> Vec<Vec<String>> {
    use std::collections::{BTreeSet, HashMap, HashSet};
    let mut d_set = DSet::new(accounts.len());
    let account_sets = accounts
        .iter()
        .map(|x| HashSet::from_iter(x[1..].iter()))
        .collect::<Vec<HashSet<&String>>>();
    for i in 0..accounts.len() - 1 {
        for j in 1..accounts.len() {
            if account_sets[i].intersection(&account_sets[j]).count() >= 1 {
                d_set.union(i, j);
            }
        }
    }
    let mut map: HashMap<usize, HashSet<&String>> = HashMap::new();
    for i in 0..accounts.len() {
        let fa = d_set.find(i);
        for mail in accounts[i][1..].iter() {
            map.entry(fa).or_default().insert(mail);
        }
    }
    map.into_iter()
        .map(|(name, mails)| {
            let mut r = vec![accounts[name][0].clone()];
            let mut a: Vec<String> = mails.into_iter().map(|x| x.clone()).collect();
            a.sort();
            r.extend_from_slice(&a[..]);
            r
        })
        .collect()
}
