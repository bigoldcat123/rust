use std::{i32, vec};
// sparse grafh
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
            head.push((-dis[next_node],next_node));
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
