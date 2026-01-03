use std::{i32, os::unix::raw::uid_t};

fn eratosthenes(n: usize) -> (Vec<bool>, Vec<usize>) {
    if n < 2 {
        return (vec![false; n + 1], Vec::new());
    }
    let mut is_prime = vec![true; n + 1];
    is_prime[0] = false;
    is_prime[1] = false;

    let mut primes = Vec::new();

    for i in 2..=n {
        if is_prime[i] {
            primes.push(i);
            // 防止 i * i 溢出：使用 checked_mul 或转为 u64
            if i as u64 * i as u64 > n as u64 {
                continue;
            }
            // 从 i*i 开始标记合数
            let mut j = i * i;
            while j <= n {
                is_prime[j] = false;
                j += i;
            }
        }
    }

    (is_prime, primes)
}
pub fn min_operations(n: i32, m: i32) -> i32 {
    use std::collections::{VecDeque,HashMap,BinaryHeap};
    let mut step = 1;
    let mut d = vec![];
    while step <= n {
        d.push(step);
        step *= 10;
    }
    let (is_prime, primes) = eratosthenes(10000);
    let mut map: HashMap<i32, Vec<i32>> = HashMap::new();
    let mut q = VecDeque::new();
    if !is_prime[n as usize] {
        q.push_back(n);
    }
    if is_prime[m as usize] {
        return -1
    }
    let mut vis = vec![false;10000];
    while !q.is_empty() {
        for n in q.split_off(0) {
            vis[n as usize] = true;

            for &d in d.iter() {
                let digit = (n / d) % 10;
                if digit > 0 && !is_prime[(n - d) as usize]{
                    if !vis[(n - d) as usize]  {
                        q.push_back(n - d);
                        vis[(n - d) as usize] = true;
                    }
                    map.entry(n).or_default().push(n - d);
                }
                if digit < 9 && !is_prime[(n + d) as usize]{
                    if !vis[(n + d) as usize] {
                        q.push_back(n + d);
                        vis[(n + d) as usize] = true;

                    }
                    map.entry(n).or_default().push(n + d);
                }
            }
        }
    }
    let mut inf = i32::MAX / 2;
    let mut dis = vec![inf;10000];
    dis[n as usize] = 0;
    let mut heap = BinaryHeap::new();
    heap.push((0,n as usize));
    while let Some((cost,n)) = heap.pop() {
        let cost = -cost;
        if cost > dis[n] {
            continue;
        }
        for &next in map.get(&(n as i32)).unwrap_or(&vec![]) {
            let new_cost = cost + next;
            if dis[next as usize] > new_cost {
                dis[next as usize] = new_cost;
                heap.push((-new_cost, next as usize));
            }
        }
    }
    dis[m as usize]
}

pub fn minimum_cost(start: Vec<i32>, target: Vec<i32>, special_roads: Vec<Vec<i32>>) -> i32 {
    use std::collections::{BinaryHeap, HashMap};
    let inf = i32::MAX / 2;
    let mut map: HashMap<(i32, i32), Vec<((i32, i32), i32)>> = HashMap::new();
    let mut dis = HashMap::new();
    for special in special_roads.iter() {
        let (x1, y1, x2, y2, cost) = (special[0], special[1], special[2], special[3], special[4]);
        map.entry((x1, y1)).or_default().push(((x2, y2), cost));
        map.entry((start[0], start[1]))
            .or_default()
            .push(((x1, y1), (start[0] - x1).abs() + (start[1] - y1).abs()));

        dis.insert((x1, y1), inf);
        dis.insert((x2, y2), inf);
        map.entry((x2, y2)).or_default().push((
            (target[0], target[1]),
            (target[0] - x2).abs() + (target[1] - y2).abs(),
        ));
    }
    for i in 0..special_roads.len() {
        for j in 0..special_roads.len() {
            if j == i {
                continue;
            }
            let (x1, y1, x2, y2) = (
                special_roads[i][2],
                special_roads[i][3],
                special_roads[j][0],
                special_roads[j][1],
            );
            map.entry((x1, y1))
                .or_default()
                .push(((x1, y1), (x1 - x2).abs() + (y1 - y2).abs()));
        }
    }
    map.entry((start[0], start[1])).or_default().push((
        (target[0], target[1]),
        (start[0] - target[0]).abs() + (start[1] - target[1]).abs(),
    ));
    dis.insert((start[0], start[1]), 0);
    dis.insert((target[0], target[1]), inf);
    let mut heap = BinaryHeap::new();
    heap.push((0, (start[0], start[1])));
    // println!("{:?}", map);
    while let Some((cost, i)) = heap.pop() {
        // println!("{:?}",heap);
        let cost = -cost;
        if cost > dis[&i] {
            continue;
        }
        for &((x2, y2), cost2) in map.get(&i).unwrap_or(&vec![]) {
            let new_cost = cost + cost2;
            // println!("{} {} {} {} {}",x2,y2,cost2,new_cost,dis[&i]);
            if new_cost < dis[&(x2, y2)] {
                *dis.entry((x2, y2)).or_default() = new_cost;
                heap.push((-new_cost, (x2, y2)));
            }
        }
    }
    // println!("{:?}", dis);
    dis[&(target[0], target[1])]
}

pub fn swim_in_water(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::BinaryHeap;

    let n = grid.len();
    let mut inf = i32::MAX / 2;

    let mut dis = vec![vec![inf; n]; n];
    let mut heap = BinaryHeap::new();
    let d = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    dis[0][0] = grid[0][0];
    heap.push((-grid[0][0], (0, 0)));
    while let Some((cost, (i, j))) = heap.pop() {
        let cost = -cost;
        if cost > dis[i][j] {
            continue;
        }
        for &(dx, dy) in &d {
            let x = i as i32 + dx;
            let y = j as i32 + dy;
            if x < 0 || x >= n as i32 || y < 0 || y >= n as i32 {
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            let new_cost = cost.max(grid[x][y]);
            if new_cost < dis[x][y] {
                dis[x][y] = new_cost;
                heap.push((-new_cost, (x, y)));
            }
        }
    }
    dis[n - 1][n - 1]
}

pub fn find_answer(n: i32, edges: Vec<Vec<i32>>) -> Vec<bool> {
    use std::collections::{BinaryHeap, HashMap, VecDeque};
    let mut map = vec![vec![]; n as usize];
    let mut edge_idx_map = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        let (u, v, w) = (e[0] as usize, e[1] as usize, e[2]);
        map[u].push((v, w));
        map[v].push((u, w));
        edge_idx_map.insert((u, v), i);
    }

    let mut inf = i32::MAX / 2;
    let mut dis = vec![inf; n as usize];
    dis[n as usize - 1] = 0;
    let mut heap = BinaryHeap::new();
    heap.push((0, n as usize - 1));
    while let Some((cost, node)) = heap.pop() {
        let cost = -cost;
        if cost > dis[node] {
            continue;
        }
        for &(next, next_cost) in map[node].iter() {
            let c = next_cost + cost;
            if c < dis[next] {
                dis[next] = c;
                heap.push((-c, next));
            }
        }
    }
    let mut ans = vec![false; n as usize];
    let mut ans2 = vec![false; edges.len()];
    let mut q = VecDeque::new();
    if dis[0] != inf {
        q.push_back(0);
        ans[0] = true;
    }
    while !q.is_empty() {
        for node in q.split_off(0) {
            for &(next_node, cost) in map[node].iter() {
                if dis[next_node] == dis[node] - cost && !ans[next_node] {
                    q.push_back(next_node);
                    ans2[*edge_idx_map.get(&(node, next_node)).unwrap()] = true;
                    ans[next_node] = true;
                }
            }
        }
    }
    ans2
}

pub fn count_restricted_paths(n: i32, edges: Vec<Vec<i32>>) -> i32 {
    use std::collections::{BinaryHeap, VecDeque};
    let mut map = vec![vec![]; n as usize];
    for e in edges {
        let (u, v, w) = (e[0] as usize, e[1] as usize, e[2]);
        map[u].push((v - 1, w));
        map[v].push((u - 1, w));
    }
    let mut inf = i32::MAX / 2;
    let mut dis = vec![inf; n as usize];
    dis[n as usize - 1] = 0;
    let mut heap = BinaryHeap::new();
    heap.push((0, n as usize - 1));
    while let Some((cost, node)) = heap.pop() {
        let cost = -cost;
        if cost > dis[node] {
            continue;
        }
        for &(next, next_cost) in map[node].iter() {
            let c = next_cost + cost;
            if c < dis[next] {
                dis[next] = c;
                heap.push((-c, next));
            }
        }
    }
    let mut q = VecDeque::new();
    let mut ans = 0;

    if dis[0] != inf {
        q.push_back(0);
    }
    while !q.is_empty() {
        for node in q.split_off(0) {
            if node == n as usize - 1 {
                ans += 1;
                continue;
            }
            for &(next_node, _) in map[node].iter() {
                if dis[node] > dis[next_node] {
                    q.push_back(next_node);
                }
            }
        }
    }
    ans
}

pub fn minimum_effort_path(heights: Vec<Vec<i32>>) -> i32 {
    use std::collections::BinaryHeap;

    let mut inf = i32::MAX / 2;

    let mut dis = vec![vec![inf; heights[0].len()]; heights.len()];
    let mut heap = BinaryHeap::new();
    let d = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    heap.push((0, (0, 0)));
    while let Some((cost, (i, j))) = heap.pop() {
        let cost = -cost;
        if cost > dis[i][j] {
            continue;
        }
        for &(dx, dy) in &d {
            let x = i as i32 + dx;
            let y = j as i32 + dy;
            if x < 0 || x >= heights.len() as i32 || y < 0 || y >= heights[0].len() as i32 {
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            let new_cost = cost.max((heights[x][y] - heights[i][j]).abs());
            if new_cost < dis[x][y] {
                dis[x][y] = new_cost;
                heap.push((-new_cost, (x, y)));
            }
        }
    }
    dis[heights.len() - 1][heights[0].len() - 1]
}

pub fn min_time_to_reach(move_time: Vec<Vec<i32>>) -> i32 {
    use std::collections::BinaryHeap;

    let mut inf = i32::MAX / 2;
    let mut dis = vec![vec![inf; move_time[0].len()]; move_time.len()];
    let mut heap = BinaryHeap::new();
    heap.push((0, (0, 0), 1));
    let d = [(1, 0), (0, 1), (-1, 0), (0, -1)];
    while let Some((cost, (i, j), prev_move_cost)) = heap.pop() {
        let cost = -cost;
        if cost > dis[i][j] {
            continue;
        }
        let next_move_cost = if prev_move_cost == 1 { 2 } else { 1 };
        for &(dx, dy) in &d {
            let (x, y) = (i as i32 + dx, j as i32 + dy);
            if x < 0 || x >= move_time.len() as i32 || y < 0 || y >= move_time[0].len() as i32 {
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            if cost >= move_time[x][y] {
                let new_cost = cost + next_move_cost;
                if new_cost < dis[x][y] {
                    dis[x][y] = new_cost;
                    heap.push((-new_cost, (x, y), next_move_cost));
                }
            } else {
                let new_cost = move_time[x][y] + next_move_cost;
                if new_cost < dis[x][y] {
                    dis[x][y] = new_cost;
                    heap.push((-new_cost, (x, y), next_move_cost));
                }
            }
        }
    }
    *dis.last().map(|x| x.last().unwrap()).unwrap()
}

pub fn min_cost(n: i32, edges: Vec<Vec<i32>>) -> i32 {
    use std::collections::BinaryHeap;
    let mut map = vec![vec![]; n as usize];

    for edge in edges {
        let u = edge[0] as usize;
        let v = edge[1] as usize;
        let cost = edge[2];
        map[u].push((v, cost));
        map[v].push((u, cost * 2));
    }
    let inf = i32::MAX / 2;
    let mut dis = vec![inf; n as usize];
    let mut heap = BinaryHeap::new();
    heap.push((0, 0));
    while let Some((cost, node)) = heap.pop() {
        let cost = -cost;
        if cost > dis[node] {
            continue;
        }
        for &(v, cost) in &map[node] {
            let new_cost = dis[node] + cost;
            if new_cost < dis[v] {
                dis[v] = new_cost;
                heap.push((-dis[v], v));
            }
        }
    }
    if dis[n as usize - 1] == inf {
        -1
    } else {
        dis[n as usize - 1]
    }
}

#[derive(PartialEq, PartialOrd)]
struct Node {
    id: usize,
    pos: f64,
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pos.partial_cmp(&other.pos).unwrap()
    }
}
impl Eq for Node {}

pub fn max_probability(
    n: i32,
    edges: Vec<Vec<i32>>,
    succ_prob: Vec<f64>,
    start_node: i32,
    end_node: i32,
) -> f64 {
    use std::collections::BinaryHeap;
    let mut map = vec![vec![]; n as usize];
    for (i, e) in edges.iter().enumerate() {
        let u = e[0];
        let v = e[1];
        let pos = succ_prob[i];
        map[u as usize].push((v as usize, pos));
        map[v as usize].push((u as usize, pos));
    }
    let mut dis = vec![0.0; n as usize];
    let mut heap = BinaryHeap::new();
    heap.push(Node {
        id: start_node as usize,
        pos: 1.0,
    });
    while let Some(node) = heap.pop() {
        if node.pos < dis[node.id] {
            continue;
        }
        if node.id == end_node as usize {
            return node.pos;
        }
        for &(v, pos) in &map[node.id] {
            let new_pos = dis[node.id] * pos;
            if new_pos > dis[v] {
                dis[v] = new_pos;
                heap.push(Node {
                    id: v,
                    pos: new_pos,
                });
            }
        }
    }
    dis[end_node as usize]
}

// sparse graph
pub fn min_time(n: i32, edges: Vec<Vec<i32>>) -> i32 {
    use std::collections::BinaryHeap;
    let mut next_nodes = vec![vec![]; n as usize];
    for e in edges {
        let u = e[0] as usize;
        let v = e[1] as usize;
        let start = e[2];
        let end = e[3];
        next_nodes[u].push((v, start, end));
    }

    let mut head = BinaryHeap::new();
    let inf = i32::MAX / 2;
    let mut dis = vec![inf; n as usize];
    dis[0] = 0;
    head.push((0, 0));
    while let Some((cost, node)) = head.pop() {
        let cost = -cost;
        if cost > dis[node] {
            continue;
        }
        for &(next_node, start, end) in &next_nodes[node] {
            if cost < start {
                dis[next_node] = dis[next_node].min(start + 1);
            } else if cost <= end {
                dis[next_node] = dis[next_node].min(cost + 1);
            }
            head.push((-dis[next_node], next_node));
        }
    }
    if dis.last().copied().unwrap() == inf {
        -1
    } else {
        dis.last().copied().unwrap()
    }
}

fn q_2642() {
    struct Graph {
        map: Vec<Vec<(usize, i32)>>, // i -> map[i][..] with cost
    }

    impl Graph {
        fn new(n: i32, edges: Vec<Vec<i32>>) -> Self {
            let mut map = vec![vec![]; n as usize];
            for e in edges {
                let u = e[0] as usize;
                let v = e[1] as usize;
                let cost = e[2];
                map[u].push((v, cost));
            }
            Self { map }
        }

        fn add_edge(&mut self, e: Vec<i32>) {
            let u = e[0] as usize;
            let v = e[1] as usize;
            let cost = e[2];
            self.map[u].push((v, cost));
        }

        fn shortest_path(&self, node1: i32, node2: i32) -> i32 {
            let len = self.map.len();
            let mut done = vec![false; len];
            let inf = i32::MAX / 2;
            let mut dis = vec![inf; len];
            dis[node1 as usize] = 0;
            loop {
                let mut min_cost = inf;
                let mut min_cost_node = 0;
                for i in 0..len {
                    if !done[i] && dis[i] < min_cost {
                        min_cost = dis[i];
                        min_cost_node = i;
                    }
                }
                if min_cost == inf {
                    return -1;
                }
                if min_cost_node == node2 as usize {
                    return min_cost;
                }
                done[min_cost_node] = true;
                for &(next, cost) in self.map[min_cost_node].iter() {
                    if dis[next] > cost + min_cost {
                        dis[next] = cost + min_cost;
                    }
                }
            }
        }
    }
}

pub fn minimum_time(n: i32, edges: Vec<Vec<i32>>, disappear: Vec<i32>) -> Vec<i32> {
    use std::collections::BinaryHeap;
    let mut map = vec![vec![]; n as usize];
    for e in edges {
        let u = e[0] as usize;
        let v = e[1] as usize;
        let cost = e[2];
        map[u].push((v, cost));
        map[v].push((u, cost));
    }
    let mut ans = vec![i32::MAX / 2; n as usize];
    let mut head = BinaryHeap::new();
    head.push((0, 0));
    while let Some((min_cost, node)) = head.pop() {
        let min_cost = -min_cost;
        if min_cost > ans[node] {
            continue;
        }
        for &(next, cost) in map[node].iter() {
            if min_cost + cost < disappear[next] && min_cost + cost < ans[next] {
                ans[next] = cost + min_cost;
                head.push((-ans[next], next));
            }
        }
    }
    ans.iter_mut().for_each(|x| {
        if *x == i32::MAX / 2 {
            *x = -1;
        }
    });

    ans
}
