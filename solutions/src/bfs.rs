use std::{
    collections::{HashMap, HashSet, VecDeque},
    i32,
};

pub fn min_cost(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::VecDeque;

    let mut dis = grid.clone();
    dis.iter_mut().for_each(|x| x.fill(i32::MAX));
    dis[0][0] = 0;
    let mut ans = 0;
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut q = VecDeque::from([(0usize, 0usize)]);
    while let Some((i, j)) = q.pop_front() {
        if i == grid.len() - 1 && j == grid[0].len() - 1 {
            return dis[i][j];
        }
        let (di, dj) = match grid[i][j] {
            1 => {
                // right
                (0, 1)
            }
            2 => {
                // left
                (0, -1)
            }
            3 => {
                // low
                (-1, 0)
            }
            _ => {
                // up
                (1, 0)
            }
        };
        let ni = i as i32 + di;
        let nj = j as i32 + dj;
        if ni >= 0
            && (ni as usize) < grid.len()
            && nj >= 0
            && (nj as usize) < grid[0].len()
            && dis[ni as usize][nj as usize] > dis[i][j]
        {
            dis[ni as usize][nj as usize] = dis[i][j];
            q.push_front((ni as usize, nj as usize));
        }
        for (di, dj) in d.iter().filter(|&&x| x != (ni, nj)) {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            if ni >= 0 && (ni as usize) < grid.len() && nj >= 0 && (nj as usize) < grid[0].len() && dis[ni as usize][nj as usize] > dis[i][j] + 1{
                dis[ni as usize][nj as usize] = dis[i][j] + 1;
                q.push_back((ni as usize, nj as usize));
            }
        }
    }

    *dis.last().unwrap().last().unwrap()
}

pub fn minimum_obstacles(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashMap, VecDeque};

    let mut ans = grid.clone();
    ans.iter_mut().for_each(|x| x.fill(i32::MAX));

    let mut q = VecDeque::from([(0, 0, grid[0][0])]);
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    while !q.is_empty() {
        for (i, j, r) in q.split_off(0) {
            if ans[i as usize][j as usize] > r {
                ans[i as usize][j as usize] = r;
            }
            let mut map: HashMap<(i32, i32), i32> = HashMap::new();
            for (di, dj) in d.iter() {
                let ni = i + di;
                let nj = j + dj;
                if ni >= 0
                    && (ni as usize) < grid.len()
                    && nj >= 0
                    && (nj as usize) < grid[0].len()
                    && ans[ni as usize][nj as usize] > r + grid[ni as usize][nj as usize]
                {
                    ans[i as usize][j as usize] = r + grid[ni as usize][nj as usize];
                    map.insert((ni, nj), r + grid[ni as usize][nj as usize]);
                }
            }
            for ((i, j), r) in map {
                q.push_back((i, j, r));
            }
        }
    }
    ans.last().unwrap().last().unwrap().clone()
}

// pub fn minimum_obstacles(grid: Vec<Vec<i32>>) -> i32 {
//     use std::collections::VecDeque;

//     let mut q = VecDeque::from([(0, 0, grid[0][0])]);
//     let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
//     let mut ans = i32::MAX;
//     while !q.is_empty() {
//         for (i, j, r) in q.split_off(0) {
//             if i as usize == grid.len() - 1 && j as usize == grid[0].len() - 1 {
//                 ans = ans.min(r);
//             }
//             for (di, dj) in d.iter() {
//                 let ni = i + di;
//                 let nj = j + dj;
//                 if ni >= 0
//                     && (ni as usize) < grid.len()
//                     && nj >= 0
//                     && (nj as usize) < grid[0].len()
//                     && ans > r + grid[ni as usize][nj as usize]
//                 {
//                     q.push_back((ni, nj, r + grid[ni as usize][nj as usize]));
//                 }
//             }
//         }
//     }
//     ans
// }

pub fn find_safe_walk(grid: Vec<Vec<i32>>, health: i32) -> bool {
    use std::collections::{HashSet, VecDeque};
    let mut vis = HashSet::from([(0, 0)]);
    let mut q = VecDeque::from([(0, 0, health)]);
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    while !q.is_empty() {
        for (i, j, h) in q.split_off(0) {
            if i as usize == grid.len() - 1 && j as usize == grid[0].len() - 1 {
                return true;
            }
            for (di, dj) in d.iter() {
                let i = i + di;
                let j = j + dj;
                if i >= 0
                    && (i as usize) < grid.len()
                    && j >= 0
                    && (j as usize) < grid[0].len()
                    && vis.insert((i, j))
                    && h - grid[i as usize][j as usize] >= 1
                {
                    q.push_back((i, j, h - -grid[i as usize][j as usize]));
                }
            }
        }
    }
    false
}

pub fn num_buses_to_destination(routes: Vec<Vec<i32>>, source: i32, target: i32) -> i32 {
    use std::collections::{HashMap, HashSet, VecDeque};
    let mut stop_buses: HashMap<i32, Vec<usize>> = HashMap::new();
    for (bus, r) in routes.iter().enumerate() {
        for &stop in r {
            stop_buses.entry(stop).or_default().push(bus);
        }
    }

    if source == target {
        return 0;
    }
    if !stop_buses.contains_key(&source) {
        return -1;
    }
    let mut vis: HashSet<usize> = HashSet::from_iter(stop_buses[&source].iter().copied());
    let mut q = VecDeque::from_iter(stop_buses[&source].iter().copied());

    let routes: Vec<HashSet<i32>> = routes
        .into_iter()
        .map(|x| HashSet::from_iter(x.into_iter()))
        .collect();

    let mut ans = 1;

    while !q.is_empty() {
        for bus in q.split_off(0) {
            if routes[bus].contains(&target) {
                return ans;
            }
            for stop in routes[bus].iter() {
                for &b in stop_buses[stop].iter() {
                    if vis.insert(b) {
                        q.push_back(b);
                    }
                }
            }
        }
        ans += 1;
    }

    ans
}

pub fn find_shortest_cycle(n: i32, edges: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut ans = i32::MAX;
    let mut map = vec![vec![]; n as usize];
    for e in edges {
        map[e[0] as usize].push(e[1] as usize);
        map[e[1] as usize].push(e[0] as usize);
    }
    for i in 0..n as usize {
        let mut q = VecDeque::from([(i, -1_i32)]);
        let mut dis = vec![-1; n as usize];
        dis[i] = 0;
        let mut step = 0;
        while !q.is_empty() {
            let (n, fa) = q.pop_front().unwrap();
            for &next in map[n].iter() {
                if dis[next] == -1 {
                    dis[next] = dis[n] + 1;
                    q.push_back((next, n as i32));
                } else if next as i32 != fa {
                    ans = ans.min(dis[next] + dis[n] + 1);
                }
            }
        }
    }
    if ans == i32::MAX { -1 } else { ans }
}

pub fn network_becomes_idle(edges: Vec<Vec<i32>>, patience: Vec<i32>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut min_dis = vec![0; patience.len()];
    let mut map = vec![vec![]; patience.len()];
    for e in edges {
        map[e[0] as usize].push(e[1] as usize);
        map[e[1] as usize].push(e[0] as usize);
    }
    let mut ans = 0;
    let mut step = 0;
    let mut q = VecDeque::from([0_usize]);
    let mut vis = HashSet::from([0_usize]);
    while !q.is_empty() {
        for n in q.split_off(0) {
            min_dis[n] = step;
            for &next in map[n].iter() {
                if vis.insert(next) {
                    q.push_back(next);
                }
            }
        }
        step += 1;
    }
    let mut ans = 0;
    for (i, d) in min_dis.into_iter().skip(1).enumerate() {
        ans = ans.max(d * 2 + (d * 2) / patience[i] - 1)
    }

    ans
}

pub fn shortest_alternating_paths(
    n: i32,
    red_edges: Vec<Vec<i32>>,
    blue_edges: Vec<Vec<i32>>,
) -> Vec<i32> {
    use std::collections::{HashSet, VecDeque};
    let mut blue_edges_map: Vec<Vec<i32>> = vec![vec![]; n as usize];
    let mut red_edges_map: Vec<Vec<i32>> = vec![vec![]; n as usize];
    for e in red_edges {
        red_edges_map[e[0] as usize].push(e[1]);
    }
    for e in blue_edges {
        blue_edges_map[e[0] as usize].push(e[1]);
    }
    let mut ans = vec![0; n as usize];
    for end in 1..ans.len() {
        let mut q = VecDeque::new();
        let mut vis = HashSet::from([('r', 0), ('b', 0)]);
        for &e in blue_edges_map[0].iter() {
            if vis.insert(('b', e)) {
                q.push_back(('b', e));
            }
        }
        for &e in red_edges_map[0].iter() {
            if vis.insert(('r', e)) {
                q.push_back(('r', e));
            }
        }

        let mut step = 1;
        let mut is_ok = true;
        while !q.is_empty() && is_ok {
            for (color, e) in q.split_off(0) {
                if e == end as i32 {
                    ans[end] = step;
                    is_ok = false;
                    break;
                }
                if color == 'r' {
                    for &e in blue_edges_map[e as usize].iter() {
                        if vis.insert(('b', e)) {
                            q.push_back(('b', e));
                        }
                    }
                } else {
                    for &e in red_edges_map[e as usize].iter() {
                        if vis.insert(('r', e)) {
                            q.push_back(('r', e));
                        }
                    }
                }
            }
            step += 1;
        }
    }

    ans
}

pub fn count_of_pairs(n: i32, x: i32, y: i32) -> Vec<i32> {
    use std::collections::{HashSet, VecDeque};
    let mut ans = vec![0; n as usize];
    for start_house in 1..=n {
        let mut step = 0;
        let mut q = VecDeque::from([start_house]);
        let mut vis = HashSet::from([start_house]);
        while !q.is_empty() {
            for h in q.split_off(0) {
                if h == 1 || h == n {
                    if h == 1 && vis.insert(h + 1) {
                        q.push_back(h + 1);
                    }
                    if h == n && vis.insert(h - 1) {
                        q.push_back(h - 1);
                    }
                } else {
                    if vis.insert(h - 1) {
                        q.push_back(h - 1);
                    }
                    if vis.insert(h + 1) {
                        q.push_back(h + 1);
                    }
                }
                if h == x || h == y {
                    if h == x && vis.insert(y) {
                        q.push_back(y);
                    }
                    if h == y && vis.insert(x) {
                        q.push_back(x);
                    }
                }
            }
            ans[step as usize] += q.len() as i32;
            step += 1;
        }
    }
    ans
}
pub fn watched_videos_by_friends(
    watched_videos: Vec<Vec<String>>,
    friends: Vec<Vec<i32>>,
    id: i32,
    level: i32,
) -> Vec<String> {
    use std::collections::{HashSet, VecDeque};

    let mut q = VecDeque::from([id as usize]);
    let mut l = 0;
    let mut vis = HashSet::from([id as usize]);
    while !q.is_empty() && l < level {
        for p in q.split_off(0) {
            for &n in friends[p].iter() {
                if vis.insert(n as usize) {
                    q.push_back(n as usize);
                }
            }
        }
        l += 1;
    }
    let mut map: HashMap<&String, i32> = std::collections::HashMap::new();
    for p in q {
        for v in watched_videos[p].iter() {
            *map.entry(v).or_default() += 1;
        }
    }
    let mut v: Vec<(&String, i32)> = map.into_iter().collect();
    v.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(b.0)));

    v.into_iter().map(|x| x.0).cloned().collect()
}

pub fn shortest_distance_after_queries(n: i32, queries: Vec<Vec<i32>>) -> Vec<i32> {
    let mut g = vec![vec![]; n as usize];
    for i in 0..g.len() - 1 {
        g[i].push(i + 1);
    }
    let mut ans = vec![];
    for (from, to) in queries.into_iter().map(|x| (x[0] as usize, x[1] as usize)) {
        g[from].push(to);
        ans.push(cal_shortest_path(&g));
    }

    ans
}
fn cal_shortest_path(g: &Vec<Vec<usize>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::from([0]);
    let mut step = 0;
    while !q.is_empty() {
        for i in q.split_off(0) {
            if i == g.len() - 1 {
                return step;
            }
            for &n in g[i].iter() {
                q.push_back(n);
            }
        }
    }
    0
}

struct InfectedArea {
    area: Vec<(usize, usize)>,
    next: i32,
}

pub fn contain_virus(mut is_infected: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::new();
    collect_virus(&mut q, &is_infected);
    let mut ans = 0;
    while !q.is_empty() {
        q.make_contiguous().sort_by_key(|x| x.next);
        let protect = q.pop_back().unwrap();
        ans += add_wall(protect, &mut is_infected);
        infect(&mut q, &mut is_infected);
        collect_virus(&mut q, &is_infected);
    }
    ans
}
fn infect(q: &mut VecDeque<InfectedArea>, is_infected: &mut Vec<Vec<i32>>) {
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    for area in q.split_off(0) {
        for (i, j) in area.area {
            for &(di, dj) in d.iter() {
                let i = i as i32 + di;
                let j = j as i32 + dj;
                if i >= 0
                    && i < is_infected.len() as i32
                    && j >= 0
                    && j < is_infected[0].len() as i32
                {
                    if is_infected[i as usize][j as usize] == 0 {
                        is_infected[i as usize][j as usize] = 1;
                    }
                }
            }
        }
    }
}
fn add_wall(protect: InfectedArea, is_infected: &mut Vec<Vec<i32>>) -> i32 {
    let mut walls = 0;
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    for (i, j) in protect.area {
        for &(di, dj) in d.iter() {
            let i = i as i32 + di;
            let j = j as i32 + dj;
            if i >= 0 && i < is_infected.len() as i32 && j >= 0 && j < is_infected[0].len() as i32 {
                if is_infected[i as usize][j as usize] == 0 {
                    walls += 1;
                }
            }
        }
        is_infected[i][j] = 2;
    }
    walls
}

fn collect_virus(q: &mut VecDeque<InfectedArea>, is_infected: &Vec<Vec<i32>>) {
    let mut vis = HashSet::new();

    let mut all_infected = true;
    for i in 0..is_infected.len() {
        for j in 0..is_infected[0].len() {
            if is_infected[i][j] == 1 {
                let mut area = InfectedArea {
                    area: Vec::new(),
                    next: 0,
                };
                dfs_collect_virus(i as i32, j as i32, &is_infected, &mut vis, &mut area);
                if !area.area.is_empty() {
                    q.push_back(area);
                }
            }
            if is_infected[i][j] == 0 {
                all_infected = false;
            }
        }
    }
    if all_infected {
        q.clear();
    }
}
fn dfs_collect_virus(
    i: i32,
    j: i32,
    is_infected: &Vec<Vec<i32>>,
    vis: &mut HashSet<(usize, usize)>,
    area: &mut InfectedArea,
) {
    if i >= 0
        && (i as usize) < is_infected.len()
        && j >= 0
        && (j as usize) < is_infected[0].len()
        && is_infected[i as usize][j as usize] == 1
        && !vis.insert((i as usize, j as usize))
    {
        area.area.push((i as usize, j as usize));
        let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
        for (di, dj) in d {
            let i = di + i;
            let j = dj + j;
            if i >= 0
                && (i as usize) < is_infected.len()
                && j >= 0
                && (j as usize) < is_infected[0].len()
                && is_infected[i as usize][j as usize] == 0
            {
                area.next += 1;
            }
        }
        dfs_collect_virus(i + 1, j, is_infected, vis, area);
        dfs_collect_virus(i - 1, j, is_infected, vis, area);
        dfs_collect_virus(i, j + 1, is_infected, vis, area);
        dfs_collect_virus(i, j - 1, is_infected, vis, area);
    }
}

pub fn maximum_safeness_factor(mut grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut thieves = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            if grid[i][j] == 1 {
                thieves.push_back((i as i32, j as i32));
                vis.insert((i as i32, j as i32));
            }
        }
    }
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    let mut r = 0;
    let mut safety = 0;
    while !thieves.is_empty() {
        for (i, j) in thieves.split_off(0) {
            grid[i as usize][j as usize] = safety;
            r = safety;
            for (di, dj) in d.iter() {
                let i = i as i32 + di;
                let j = j as i32 + dj;
                if i >= 0
                    && (i as usize) < grid.len()
                    && j >= 0
                    && (j as usize) < grid[0].len()
                    && vis.insert((i, j))
                {
                    thieves.push_back((i, j));
                }
            }
        }
        safety += 1;
    }

    let mut l = 0;
    while l <= r {
        let mid = (r + l) / 2;
        if check(mid, &grid) {
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }
    l
}

fn check(max: i32, grid: &Vec<Vec<i32>>) -> bool {
    if grid[0][0] < max {
        return false;
    }
    let mut p = VecDeque::from([(0, 0)]);
    let mut vis = HashSet::from([(0, 0)]);
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    while !p.is_empty() {
        for (i, j) in p.split_off(0) {
            if i == grid.len() - 1 && j == grid[0].len() - 1 {
                return true;
            }
            for (di, dj) in d.iter() {
                let i = i as i32 + di;
                let j = j as i32 + dj;
                if i >= 0
                    && (i as usize) < grid.len()
                    && j >= 0
                    && (j as usize) < grid[0].len()
                    && vis.insert((i, j))
                    && grid[i as usize][j as usize] >= max
                {
                    p.push_back((i as usize, j as usize));
                }
            }
        }
    }
    false
}

pub fn cut_off_tree(forest: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::from([(0, 0)]);
    let mut vis = HashSet::from([(0, 0)]);

    let mut trees = vec![(0, 0, -1)];
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    while !q.is_empty() {
        for (i, j) in q.split_off(0) {
            if forest[i][j] > 1 {
                trees.push((i as i32, j as i32, forest[i][j]));
            }
            for (di, dj) in d.iter() {
                let i = i as i32 + di;
                let j = j as i32 + dj;
                if i >= 0
                    && (i as usize) < forest.len()
                    && j >= 0
                    && (j as usize) < forest[0].len()
                    && vis.insert((i, j))
                    && forest[i as usize][j as usize] > 0
                {
                    q.push_back((i as usize, j as usize));
                }
            }
        }
    }
    if forest.iter().flatten().filter(|&&x| x > 1).count() != trees.len() - 1 {
        return -1;
    }
    trees.sort_by_key(|x| x.2);

    let mut ans = 0;
    for i in 0..trees.len() - 1 {
        ans += find_next(
            &forest,
            (trees[i].0, trees[i].1),
            (trees[i + 1].0, trees[i + 1].1),
        )
    }
    ans
}
fn find_next(forest: &Vec<Vec<i32>>, from: (i32, i32), to: (i32, i32)) -> i32 {
    let mut steps = 0;
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::from([from]);
    let mut vis = HashSet::from([from]);
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    while !q.is_empty() {
        for (i, j) in q.split_off(0) {
            if to == (i, j) {
                return steps;
            }
            for (di, dj) in d.iter() {
                let i = i as i32 + di;
                let j = j as i32 + dj;
                if i >= 0
                    && (i as usize) < forest.len()
                    && j >= 0
                    && (j as usize) < forest[0].len()
                    && vis.insert((i, j))
                    && forest[i as usize][j as usize] > 0
                {
                    q.push_back((i, j));
                }
            }
        }
    }
    steps
}

pub fn shortest_path(grid: Vec<Vec<i32>>, k: i32) -> i32 {
    use std::collections::{HashSet, VecDeque};

    let mut qs = vec![(VecDeque::new()); k as usize + 1];
    qs[k as usize].push_back((0_i32, 0_i32));
    let mut vis = HashSet::new();
    vis.insert((0_i32, 0_i32));
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut ans = 0;
    while qs.iter().any(|x| !x.is_empty()) {
        println!("{:?}", qs);
        for q_i in 0..qs.len() {
            let pq = qs[q_i].split_off(0);
            for (i, j) in pq {
                if i as usize == grid.len() - 1 && j as usize == grid[0].len() - 1 {
                    return ans;
                }
                for (di, dj) in d.iter() {
                    let i = di + i;
                    let j = dj + j;
                    if i >= 0
                        && (i as usize) < grid.len()
                        && j >= 0
                        && (j as usize) < grid[0].len()
                        && vis.insert((i, j))
                    {
                        if grid[i as usize][j as usize] == 0 {
                            qs[q_i].push_back((i, j));
                        } else if q_i > 0 {
                            qs[q_i - 1].push_back((i, j));
                        }
                    }
                }
            }
        }
        ans += 1;
    }
    -1
}

pub fn highest_ranked_k_items(
    grid: Vec<Vec<i32>>,
    pricing: Vec<i32>,
    start: Vec<i32>,
    k: i32,
) -> Vec<Vec<i32>> {
    use std::collections::{HashSet, VecDeque};
    let mut ans = vec![];
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    q.push_back((start[0], start[1]));
    vis.insert((start[0], start[1]));
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut des = 0;
    while !q.is_empty() {
        q.make_contiguous().sort_by(|&(i1, j1), &(i2, j2)| {
            grid[i1 as usize][j1 as usize]
                .cmp(&grid[i2 as usize][j2 as usize])
                .then(i1.cmp(&i2))
                .then(j1.cmp(&j2))
                .reverse()
        });
        for (i, j) in q.split_off(0) {
            for &(di, dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0
                    && (i as usize) < grid.len()
                    && j >= 0
                    && (j as usize) < grid[0].len()
                    && vis.insert((i, j))
                    && grid[i as usize][j as usize] != 0
                {
                    if ans.len() == k as usize {
                        return ans;
                    }
                    if grid[i as usize][j as usize] != 1
                        && grid[i as usize][j as usize] >= pricing[0]
                        && grid[i as usize][j as usize] <= pricing[1]
                    {
                        ans.push(vec![i, j]);
                    }
                }
            }
        }
        des += 1;
    }
    ans
}

pub fn shortest_bridge(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            if grid[i][j] == 1 {
                dfs(i as i32, j as i32, &grid, &mut q, &mut vis);
            }
        }
        if !q.is_empty() {
            break;
        }
    }
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut ans = 0;
    while !q.is_empty() {
        for (i, j) in q.split_off(0) {
            for &(di, dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0
                    && (i as usize) < grid.len()
                    && j >= 0
                    && (j as usize) < grid[0].len()
                    && vis.insert((i, j))
                {
                    if grid[i as usize][j as usize] == 1 {
                        return ans;
                    }
                    q.push_back((i, j));
                }
            }
        }
        ans += 1;
    }
    ans
}

fn dfs(
    i: i32,
    j: i32,
    grid: &Vec<Vec<i32>>,
    q: &mut VecDeque<(i32, i32)>,
    vis: &mut HashSet<(i32, i32)>,
) {
    if i >= 0
        && (i as usize) < grid.len()
        && j >= 0
        && (j as usize) < grid[0].len()
        && vis.insert((i, j))
        && grid[i as usize][j as usize] == 1
    {
        q.push_back((i, j));
        let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
        for (di, dj) in d {
            dfs(i + di, j + dj, grid, q, vis);
        }
    }
}

pub fn highest_peak(mut is_water: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    use std::collections::{HashSet, VecDeque};

    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..is_water.len() {
        for j in 0..is_water[0].len() {
            q.push_back((i as i32, j as i32));
            vis.insert((i as i32, j as i32));
        }
    }
    let mut h = 0;
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    while !q.is_empty() {
        for (i, j) in q.split_off(0) {
            is_water[i as usize][j as usize] = h;
            for &(di, dj) in d.iter() {
                let i = i + di;
                let j = j + dj;
                if i >= 0
                    && (i as usize) < is_water.len()
                    && j >= 0
                    && (j as usize) < is_water[0].len()
                    && vis.insert((i, j))
                {
                    q.push_back((i, j));
                }
            }
        }
        h += 1;
    }

    is_water
}
pub fn oranges_rotting(mut mat: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    let mut ans = 0;
    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            if mat[i][j] == 2 {
                q.push_back((i as i32, j as i32));
                vis.insert((i as i32, j as i32));
            }
        }
    }

    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    while !q.is_empty() {
        let pq = q.split_off(0);
        for (i, j) in pq {
            for &(di, dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0
                    && (i as usize) < mat.len()
                    && j >= 0
                    && (j as usize) < mat[0].len()
                    && mat[i as usize][j as usize] == 1
                    && vis.insert((i, j))
                {
                    mat[i as usize][j as usize] = 2;
                    q.push_back((i, j));
                }
            }
        }
        ans += 1;
    }
    if mat.iter().flatten().all(|&x| x == 0 || x == 2) {
        ans
    } else {
        -1
    }
}

pub fn update_matrix(mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    use std::collections::{HashSet, VecDeque};
    let mut ans = vec![vec![0; mat[0].len()]; mat.len()];
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            if mat[i][j] == 0 {
                q.push_back((i as i32, j as i32));
                vis.insert((i as i32, j as i32));
            }
        }
    }
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    while !q.is_empty() {
        let pq = q.split_off(0);
        for (i, j) in pq {
            let n = ans[i as usize][j as usize];
            for &(di, dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0
                    && (i as usize) < mat.len()
                    && j >= 0
                    && (j as usize) < mat[0].len()
                    && vis.insert((i, j))
                {
                    ans[i as usize][j as usize] = n + 1;
                    q.push_back((i, j));
                }
            }
        }
    }

    ans
}

pub fn count_mentions(number_of_users: i32, mut events: Vec<Vec<String>>) -> Vec<i32> {
    use std::collections::BTreeSet;
    events.sort_by(|a, b| {
        a[1].parse::<i32>()
            .unwrap()
            .cmp(&b[1].parse::<i32>().unwrap())
            .then(a[0].cmp(&b[0]).reverse())
    });

    let mut is_online = vec![true; number_of_users as usize];
    let mut set = BTreeSet::new(); // (time_stamp,id)
    let mut mentions = vec![0; number_of_users as usize];

    for e in events {
        let time_stamp = e[1].parse::<i32>().unwrap();
        match e[0].as_str() {
            "OFFLINE" => {
                let off_line_id = e[2].parse::<usize>().unwrap();
                is_online[off_line_id] = false;
                set.insert((time_stamp + 60, off_line_id));
            }
            "MESSAGE" => match e[2].as_str() {
                "ALL" => {
                    mentions.iter_mut().for_each(|x| *x += 1);
                }
                "HERE" => {
                    while let Some((time_to_online, p_id)) = set.first() {
                        if time_stamp >= *time_to_online {
                            is_online[*p_id] = true;
                            set.pop_first();
                        }
                    }
                    mentions
                        .iter_mut()
                        .zip(is_online.iter())
                        .for_each(|(p, &is_online)| {
                            if is_online {
                                *p += 1;
                            }
                        });
                }
                other => {
                    for id in other.split_whitespace() {
                        let id = id[2..].parse::<usize>().unwrap();
                        mentions[id] += 1;
                    }
                }
            },
            _ => {
                unreachable!()
            }
        }
    }
    mentions
}
pub fn max_distance(grid: Vec<Vec<i32>>) -> i32 {
    let mut ones = vec![];
    let mut zeroes = vec![];
    for i in 0..grid.len() {
        for j in 0..grid.len() {
            if grid[i][j] == 0 {
                zeroes.push((i as i32, j as i32));
            } else {
                ones.push((i as i32, j as i32));
            }
        }
    }
    if ones.is_empty() || zeroes.is_empty() {
        return -1;
    }
    let mut ans = 0;
    for (i, j) in zeroes {
        let mut min = i32::MAX;
        for &(ii, jj) in ones.iter() {
            min = min.min((i - ii).abs() + (j - jj).abs());
        }
        ans = ans.max(min);
    }
    ans
}
pub fn shortest_path_binary_matrix(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut visited = HashSet::new();
    let mut q = VecDeque::new();
    if grid[0][0] == 0 {
        visited.insert((0_i32, 0_i32));
        q.push_back((0_i32, 0_i32));
    }
    let mut path = 0;
    let x: Vec<(i32, i32)> = vec![
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, -1),
        (-1, 1),
    ];

    while !q.is_empty() {
        let mut pq = q.split_off(0);
        for (i, j) in pq {
            if i as usize == grid.len() - 1 && j as usize == grid[0].len() - 1 {
                return path;
            }
            for &(ii, jj) in x.iter() {
                let (i, j) = (i + ii, j + jj);
                if !(i < 0 || j < 0 || i as usize >= grid.len() || j as usize >= grid[0].len()) {
                    if grid[i as usize][j as usize] == 0 && visited.insert((i, j)) {
                        q.push_back((i, j));
                    }
                }
            }
        }
        path += 1;
    }
    -1
}

pub fn nearest_exit(maze: Vec<Vec<char>>, entrance: Vec<i32>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut visited = HashSet::new();
    let mut q = VecDeque::new();
    let x = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    let i = entrance[0];
    let j = entrance[1];
    for (ii, jj) in x {
        let (i, j) = (i + ii, j + jj);
        if !(i < 0 || j < 0 || i as usize >= maze.len() || j as usize >= maze[0].len()) {
            if maze[i as usize][j as usize] != '+' {
                if visited.insert((i, j)) {
                    q.push_back((i, j));
                }
            }
        }
    }
    let mut step = -1;
    while !q.is_empty() {
        let mut pq = q.split_off(0);
        for (i, j) in pq {
            if i < 0 || j < 0 || i as usize >= maze.len() || j as usize >= maze[0].len() {
                return step;
            }
            if maze[i as usize][j as usize] != '+' {
                let x = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
                for (ii, jj) in x {
                    let next = (i + ii, j + jj);
                    if visited.insert(next) {
                        q.push_back(next);
                    }
                }
            }
        }
        step += 1;
    }
    step
}
