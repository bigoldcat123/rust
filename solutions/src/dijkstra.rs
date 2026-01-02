
pub fn find_answer(n: i32, edges: Vec<Vec<i32>>) -> Vec<bool> {
    use std::collections::{BinaryHeap, VecDeque,HashMap};
    let mut map = vec![vec![]; n as usize];
    let mut edge_idx_map = HashMap::new();
    for (i,e) in edges.iter().enumerate() {
        let (u, v, w) = (e[0] as usize, e[1] as usize, e[2]);
        map[u].push((v, w));
        map[v].push((u, w));
        edge_idx_map.insert((u,v),i);

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
    let mut ans = vec![false;n as usize];
    let mut ans2 = vec![false;edges.len()];
    let mut q = VecDeque::new();
    if dis[0] != inf {
        q.push_back(0);
        ans[0] = true;
    }
    while !q.is_empty() {
        for node in q.split_off(0) {
            for &(next_node,cost) in map[node].iter() {
                if dis[next_node] == dis[node] - cost && !ans[next_node] {
                    q.push_back(next_node);
                    ans2[*edge_idx_map.get(&(node,next_node)).unwrap()] = true;
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
