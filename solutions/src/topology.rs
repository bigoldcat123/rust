use std::{
    collections::{HashMap, HashSet, VecDeque},
    i32, vec,
};

pub fn min_time_to_reach(move_time: Vec<Vec<i32>>) -> i32 {
    let len = move_time.len() * move_time[0].len();
    let c_len = move_time[0].len() as i32;
    let mut dis = vec![i32::MAX; len];
    let mut done = vec![false; len];
    dis[0] = 0;
    loop {
        let mut min_cost = i32::MAX;
        let mut node = 0;
        for (i, &c) in dis.iter().enumerate() {
            if !done[i] && min_cost > c {
                min_cost = c;
                node = i;
            }
        }
        done[node] = true;
        let d = [-c_len, c_len, -1, 1];
        for d in d {
            let next_node = node as i32 + d;
            if next_node >= 0 && (next_node as usize) < dis.len() {
                if d.abs() == 1 && next_node as usize / c_len as usize != node / c_len as usize {
                    continue;
                }

                let next_cost = move_time[next_node as usize / c_len as usize]
                    [next_node as usize % c_len as usize];
                dis[next_node as usize] = dis[next_node as usize].min(min_cost.max(next_cost) + 1);
            }
        }
        if done.iter().all(|&x| x) {
            break;
        }
    }

    dis.last().copied().unwrap()
}

pub fn network_delay_time(times: Vec<Vec<i32>>, n: i32, k: i32) -> i32 {
    let n = n + 1;
    let mut dis = vec![i32::MAX / 2; n as usize];
    let mut m = vec![vec![]; n as usize];
    for t in times {
        m[t[0] as usize].push((t[1] as usize, t[2]));
    }
    let mut done = vec![false; n as usize];
    dis[k as usize] = 0;
    let mut ans = 0;
    loop {
        let mut min_dis = i32::MAX;
        let mut node = 0;
        for (i, &d) in dis.iter().enumerate() {
            if !done[i] && min_dis < d {
                min_dis = d;
                node = i;
            }
        }
        if min_dis == i32::MAX {
            return -1;
        }
        ans = ans.max(min_dis);
        done[node] = true;
        for &(next_node, cost) in m[node].iter() {
            dis[next_node] = dis[next_node].min(min_dis + cost);
        }
        if done.iter().all(|&x| x) {
            break;
        }
    }
    ans
}
pub fn find_duplicate(nums: Vec<i32>) -> i32 {
    use std::collections::VecDeque;
    let mut in_de = vec![0; nums.len()];
    for n in nums.iter().copied() {
        in_de[n as usize] += 1;
    }
    let mut q = VecDeque::new();
    for (i, &i_d) in in_de.iter().enumerate() {
        if i_d == 0 {
            q.push_back(i);
        }
    }
    while !q.is_empty() {
        for i in q.split_off(0) {
            in_de[nums[i] as usize] -= 1;
            if in_de[nums[i] as usize] == 0 {
                q.push_back(nums[i] as usize);
            }
        }
    }
    println!("{:?}", in_de);
    let mut ans = 0;

    ans
}

pub fn count_visited_nodes(edges: Vec<i32>) -> Vec<i32> {
    use std::collections::VecDeque;
    let mut in_degree = vec![0; edges.len()];
    for &e in edges.iter() {
        in_degree[e as usize] += 1;
    }
    let mut q = VecDeque::new();
    for (n, &in_d) in in_degree.iter().enumerate() {
        if in_d == 0 {
            q.push_back(n);
        }
    }
    let mut r = vec![vec![]; edges.len()];

    while !q.is_empty() {
        for i in q.split_off(0) {
            r[edges[i] as usize].push(i);
            in_degree[edges[i] as usize] -= 1;
            if in_degree[edges[i] as usize] == 0 {
                q.push_back(edges[i] as usize);
            }
        }
    }
    let mut ans = vec![0; edges.len()];
    for i in 0..edges.len() {
        if in_degree[i] == 0 {
            continue;
        }
        let mut nodes_in_cycle = vec![i];
        let mut n = edges[i] as usize;
        in_degree[i] = 0;
        let mut size = 1;
        while n != i {
            size += 1;
            nodes_in_cycle.push(n);
            in_degree[n] = 0;
            n = edges[n] as usize;
        }
        for n in nodes_in_cycle {
            ans[n] = size as i32;
            cal_chain(n, &r, size, &mut ans);
        }
    }
    ans
}
fn cal_chain(n: usize, r: &Vec<Vec<usize>>, size: i32, ans: &mut Vec<i32>) {
    for &child in &r[n] {
        ans[child] = size + 1;
        cal_chain(child, r, size + 1, ans);
    }
}

pub fn find_redundant_directed_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::{HashSet, VecDeque};
    let len = edges.len();
    let mut out_degree = vec![0; len];
    let mut in_degree = vec![0; len];
    let mut fa = vec![0; len];
    let mut son_ = vec![vec![]; len];
    let mut maybe = vec![];
    for e in edges.iter() {
        let son = e[1] as usize - 1;
        let f = e[0] as usize - 1;
        out_degree[f] += 1;
        in_degree[son] += 1;
        if in_degree[son] == 2 {
            maybe = e.clone();
        }
        fa[son] = f;
        son_[f].push(son);
    }
    let point_to_root = in_degree.iter().all(|&x| x != 0);

    // 1. point to root, eg. there is no node with in_degree == 0
    // there must be a cycle.
    if point_to_root {
        let mut q = VecDeque::new();
        for (i, &o) in out_degree.iter().enumerate() {
            if o == 0 {
                q.push_back(i);
            }
        }
        while let Some(n) = q.pop_front() {
            out_degree[fa[n]] -= 1;
            if out_degree[fa[n]] == 0 {
                q.push_back(fa[n]);
            }
        }
        let mut cand = HashSet::new();
        for i in 0..len {
            if out_degree[i] != 0 {
                cand.insert(vec![fa[i] as i32 + 1, i as i32 + 1]);
            }
        }
        for e in edges.into_iter().rev() {
            if cand.contains(&e) {
                return e;
            }
        }
    }
    // 2. potin to child node.
    //    1. there is a cycle
    //    2. there is no cycle, delete the last edge making any in_degree to be 2
    else {
        let mut q = VecDeque::new();
        for (i, &in_d) in in_degree.iter().enumerate() {
            if in_d == 0 {
                q.push_back(i);
            }
        }
        let mut in2 = vec![0; len];
        let mut xxx = HashSet::new();
        while let Some(n) = q.pop_front() {
            // if son_[n] < 0 {
            //     continue;
            // }
            for s in son_[n].iter().copied() {
                // let s = son_[n] as usize;
                xxx.insert(vec![n as i32 + 1, s as i32 + 1]);
                in2[s] += 1;
                in_degree[s] -= 1;
                if in_degree[s] == 0 {
                    q.push_back(s);
                }
            }
        }
        println!("{:?}", in_degree);
        // 2. there is no cycle, delete the last edge making any in_degree to be 2
        if in_degree.iter().all(|&x| x == 0) {
            println!("{}", "eee");
            return maybe;
        }
        //         let mut xxx = HashSet::new();
        //         let mut last = vec![];
        // while let Some(n) = q.pop_front() {
        //     if son_[n] < 0 {
        //         continue;
        //     }
        //     let s = son_[n] as usize;

        //     in2[s] += 1;
        //     in_degree[s] -= 1;
        //     last = vec![n as i32 + 1,s as i32 + 1];
        //     if !xxx.insert(s) {
        //         break
        //     }
        //     q.push_back(s);
        // }
        // return last
        //         println!("{:?}",xxx);
        // println!("{:?}",n2);
        for e in edges.into_iter().filter(|x| !xxx.contains(x)) {
            let f = e[0] as usize - 1;
            let s = e[1] as usize - 1;
            in2[s] += 1;
            if in2[s] == 2 {
                return e;
            }
        }
    }
    unreachable!()
}

struct DSet {
    fa: Vec<usize>,
}
impl DSet {
    fn new(size: usize) -> Self {
        Self {
            fa: (0..size).into_iter().collect(),
        }
    }
    fn get_parent(&mut self, node: usize) -> usize {
        if self.fa[node] == node {
            return node;
        } else {
            self.fa[node] = self.get_parent(self.fa[node]);
        }
        self.fa[node]
    }
    fn union(&mut self, from: usize, to: usize) -> bool {
        let x = self.get_parent(from);
        let y = self.get_parent(to);
        if x == y {
            return false;
        }
        self.fa[x] = y;
        true
    }
}
pub fn find_redundant_connection(edges: Vec<Vec<i32>>) -> Vec<i32> {
    let mut max = edges.iter().flatten().max().copied().unwrap() as usize;
    let mut d_set = DSet::new(max);
    for e in edges {
        let x = e[0] as usize - 1;
        let y = e[1] as usize - 1;
        if !d_set.union(x, y) {
            return e;
        }
    }
    unreachable!()
}

pub fn longest_cycle(edges: Vec<i32>) -> i32 {
    use std::collections::VecDeque;
    let mut in_degree = vec![0; edges.len()];
    for &e in edges.iter() {
        in_degree[e as usize] += 1;
    }
    let mut q = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            q.push_back(i);
        }
    }
    while let Some(n) = q.pop_front() {
        let next = edges[n] as usize;
        in_degree[next] -= 1;
        if in_degree[next] == 0 {
            q.push_back(next);
        }
    }

    let mut max_ring_size = 0;

    for i in 0..edges.len() {
        if in_degree[i] == 0 {
            continue;
        }
        let mut size = 1;
        let mut next = edges[i] as usize;
        in_degree[i] = 0;
        while next != i {
            in_degree[next] = 0;
            size += 1;
            next = edges[next] as usize;
        }
        max_ring_size = max_ring_size.max(size);
    }

    max_ring_size
}

pub fn closest_meeting_node(edges: Vec<i32>, node1: i32, node2: i32) -> i32 {
    let dis_1 = cal_dis(&edges, node1 as usize);
    let dis_2 = cal_dis(&edges, node2 as usize);
    let mut ans = -1;
    let mut min = edges.len() as i32;
    for i in 0..dis_1.len() {
        let max = dis_1[i].max(dis_2[i]);
        if max < min {
            min = max;
            ans = i as i32;
        }
    }
    ans
}
fn cal_dis(edges: &Vec<i32>, node: usize) -> Vec<i32> {
    let mut dis = vec![edges.len() as i32; edges.len()];
    dis[node] = 0;
    let mut d = 1;
    let mut n = edges[node];
    while n != node as i32 && n > 0 && dis[n as usize] == edges.len() as i32 {
        dis[n as usize] = d;
        n = edges[n as usize];
        d += 1;
    }

    dis
}

pub fn maximum_invitations(favorite: Vec<i32>) -> i32 {
    use std::collections::VecDeque;
    // 1. topology

    let mut in_degree = vec![0; favorite.len()];
    for &f in favorite.iter() {
        in_degree[f as usize] += 1;
    }

    // 2. reversed graph
    //
    let mut rg = vec![vec![]; favorite.len()];
    let mut q = VecDeque::new();
    for (i, &i_d) in in_degree.iter().enumerate() {
        if i_d == 0 {
            q.push_back(i);
        }
    }
    while !q.is_empty() {
        for e in q.split_off(0) {
            rg[favorite[e] as usize].push(e);
            in_degree[favorite[e] as usize] -= 1;
            if in_degree[favorite[e] as usize] == 0 {
                q.push_back(favorite[e] as usize);
            }
        }
    }
    fn dfs_chain(node: usize, rg: &Vec<Vec<usize>>) -> i32 {
        let mut ans = 1;
        let mut d = 0;
        for &next in rg[node].iter() {
            d = d.max(dfs_chain(next, rg));
        }
        ans + d
    }
    let mut max_ring = 0;
    let mut max_chain = 0;
    for i in 0..favorite.len() {
        if in_degree[i] == 0 {
            continue;
        }
        let mut ring = 0;
        in_degree[i] = 0;
        let mut x = favorite[i] as usize;
        let mut size = 1;
        while x != i {
            in_degree[x] = 0;
            size += 1;
            x = favorite[x] as usize;
        }
        if size == 2 {
            max_chain =
                max_chain.max(dfs_chain(i, &rg).max(dfs_chain(favorite[i] as usize, &rg)) + 2);
        } else {
            max_ring = max_ring.max(size)
        }
    }
    max_chain.max(max_ring)
}

pub fn minimum_time(n: i32, relations: Vec<Vec<i32>>, time: Vec<i32>) -> i32 {
    // use std::collections::VecDeque
    let mut out_degree = vec![0; n as usize];
    let mut fa = vec![vec![]; n as usize];
    for r in relations {
        out_degree[r[0] as usize - 1] += 1;
        fa[r[1] as usize - 1].push(r[0] as usize - 1);
    }
    let mut final_n = vec![];
    let mut finisn_time = vec![0; n as usize];
    for (i, &o) in out_degree.iter().enumerate() {
        if o == 0 {
            final_n.push(i);
        }
    }
    dfs_cal(final_n, &fa, &mut out_degree, &time, &mut finisn_time)
}
fn dfs_cal(
    nodes: Vec<usize>,
    fa: &[Vec<usize>],
    out_degree: &mut Vec<i32>,
    time: &Vec<i32>,
    finisn_time: &mut Vec<i32>,
) -> i32 {
    let mut ans1 = 0;
    for n in nodes.iter().copied() {
        if fa[n].is_empty() {
            finisn_time[n] = time[n];
        } else {
            let mut final_n = vec![];
            for f in fa[n].iter().copied() {
                out_degree[f] -= 1;
                if out_degree[f] == 0 {
                    final_n.push(f);
                }
            }
            if !final_n.is_empty() {
                dfs_cal(final_n, fa, out_degree, time, finisn_time);
            }
        }
    }

    for n in nodes.iter().copied() {
        let mut max = 0;
        for f in fa[n].iter().copied() {
            max = max.max(finisn_time[f]);
        }
        finisn_time[n] = max + time[n];
    }
    let mut ans1 = 0;
    for n in nodes {
        ans1 = ans1.max(finisn_time[n]);
    }
    ans1
}

pub fn loud_and_rich(richer: Vec<Vec<i32>>, quiet: Vec<i32>) -> Vec<i32> {
    use std::collections::VecDeque;

    let mut q = VecDeque::new();
    let mut fa = vec![vec![]; quiet.len()];
    let mut next_nodes = vec![vec![]; quiet.len()];
    let mut out_degree = vec![0; quiet.len()];
    for r in richer {
        fa[r[0] as usize].push(r[1] as usize);
        next_nodes[r[1] as usize].push(r[0] as usize);
        out_degree[r[1] as usize] += 1;
    }
    for (i, &o) in out_degree.iter().enumerate() {
        if o == 0 {
            q.push_back(i);
        }
    }
    println!("{:?}", next_nodes);
    let mut ans = quiet.clone();
    let mut ans2 = (0..quiet.len() as i32).into_iter().collect::<Vec<i32>>();
    while !q.is_empty() {
        println!("{:?}", q);

        for node in q.split_off(0) {
            let mut node_quite = quiet[node];
            let mut quite_node = node;
            for next in next_nodes[node].iter().copied() {
                if ans[next] < node_quite {
                    node_quite = ans[next];
                    quite_node = ans2[next] as usize;
                }
            }
            ans[node] = node_quite as i32;
            ans2[node] = quite_node as i32;
            for f in fa[node].iter().copied() {
                out_degree[f] -= 1;
                if out_degree[f] == 0 {
                    q.push_back(f);
                }
            }
        }
    }
    ans2
}
// every node only has one in_edge
// zero in_edge onley one
// at most two out_edge
pub fn validate_binary_tree_nodes(n: i32, left_child: Vec<i32>, right_child: Vec<i32>) -> bool {
    let mut in_degree = vec![0; n as usize];
    let mut out_degree = vec![0; n as usize];
    let mut f = vec![-1; n as usize];
    for i in 0..n as usize {
        if left_child[i] != -1 {
            in_degree[left_child[i] as usize] += 1;
            out_degree[i] += 1;
            f[left_child[i] as usize] = i as i32;
        }
        if right_child[i] != -1 {
            in_degree[right_child[i] as usize] += 1;
            out_degree[i] += 1;
            f[right_child[i] as usize] = i as i32;
        }
    }
    let mut q = VecDeque::new();
    for (i, o) in out_degree.iter().enumerate() {
        if *o == 0 {
            q.push_back(i);
        }
    }
    let mut remain = n;
    while !q.is_empty() {
        let mut size = q.len();
        remain -= size as i32;
        for _ in 0..size {
            let cur = q.pop_front().unwrap();
            if f[cur] != -1 {
                out_degree[f[cur] as usize] -= 1;
                if out_degree[f[cur] as usize] == 0 {
                    q.push_back(f[cur] as usize);
                }
            }
        }
    }
    remain == 0
}

pub fn find_min_height_trees(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::{HashSet, VecDeque};
    if n == 1 {
        return vec![0];
    }
    let mut degree = vec![0; n as usize];
    let mut next_nodes = vec![vec![]; n as usize];
    for e in edges {
        degree[e[0] as usize] += 1;
        degree[e[1] as usize] += 1;
        next_nodes[e[0] as usize].push(e[1]);
        next_nodes[e[1] as usize].push(e[0]);
    }
    let mut q = VecDeque::new();
    let mut unused: HashSet<usize> = HashSet::from_iter((0..n as usize).into_iter());
    for (i, &d) in degree.iter().enumerate() {
        if d == 1 {
            q.push_back(i);
            unused.remove(&i);
        }
    }
    if unused.is_empty() {
        return q.into_iter().map(|x| x as i32).collect();
    }
    while !q.is_empty() {
        for n in q.split_off(0) {
            for x in next_nodes[n]
                .iter()
                .filter(|&&x| unused.contains(&(x as usize)))
            {
                degree[*x as usize] -= 1;
            }
        }
        for (i, &d) in degree.iter().enumerate() {
            if unused.contains(&i) {
                if d <= 1 {
                    q.push_back(i);
                    unused.remove(&i);
                }
            }
        }
        let mut rm = HashSet::new();
        for i in unused.iter() {
            if degree[*i as usize] <= 1 {
                rm.insert(*i);
                q.push_back(*i);
            }
        }
        unused.retain(|&x| !rm.contains(&x));
        for i in rm {
            unused.remove(&i);
        }
        if unused.is_empty() {
            return q.into_iter().map(|x| x as i32).collect();
        }
    }

    vec![]
}

pub fn is_printable(target_grid: Vec<Vec<i32>>) -> bool {
    let mut diff = HashSet::new();
    target_grid.iter().flatten().for_each(|&x| {
        diff.insert(x);
    });
    let mut rc_map: HashMap<i32, ((usize, usize), (usize, usize))> = HashMap::new();
    for i in 0..target_grid.len() {
        for j in 0..target_grid[0].len() {
            let color = target_grid[i][j];
            if let Some(rc) = rc_map.get_mut(&color) {
                if rc.0.0 > i {
                    //row min
                    rc.0.0 = i;
                }
                if rc.0.1 < i {
                    // row max
                    rc.0.1 = i;
                }
                if rc.1.0 > j {
                    // col min
                    rc.1.0 = j;
                }
                if rc.1.1 < j {
                    // cl max
                    rc.1.1 = j;
                }
            } else {
                rc_map.insert(color, ((i, i), (j, j)));
            }
        }
    }
    for (color, ((r_min, r_max), (c_min, c_max))) in rc_map.iter() {
        println!(
            "Color: {}, Row Range: ({}, {}), Column Range: ({}, {})",
            color, r_min, r_max, c_min, c_max
        );
    }

    let mut pre = HashSet::new();
    while pre.len() != diff.len() {
        let x = find_rectangular(&target_grid, &rc_map, &pre);
        if x.len() == 0 {
            return false;
        }
        pre.extend(x.iter());
    }

    true
}
fn find_rectangular(
    target_grid: &Vec<Vec<i32>>,
    rc_map: &HashMap<i32, ((usize, usize), (usize, usize))>,
    pre: &HashSet<i32>,
) -> Vec<i32> {
    use std::collections::{HashMap, HashSet};
    let mut vis: HashSet<(usize, usize)> = HashSet::new();
    let mut res = vec![];
    for (color, (row_range, col_range)) in rc_map.iter() {
        let mut ok = true;
        for i in row_range.0..=row_range.1 {
            for j in col_range.0..=col_range.1 {
                if target_grid[i][j] != *color && !pre.contains(color) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            res.push(*color);
        }
    }
    return res;
}

pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::{HashMap, VecDeque};
    let mut out_degree = vec![0; graph.len()];
    let mut in_edges = vec![vec![]; graph.len()];
    for (outi, out) in graph.iter().enumerate() {
        out_degree[outi] = out.len() as i32;
        for &o in out {
            in_edges[o as usize].push(outi);
        }
    }
    let mut q = VecDeque::new();
    for (o, out_degree) in out_degree.iter_mut().enumerate() {
        if *out_degree == 0 {
            q.push_back(o);
            *out_degree = i32::MAX;
        }
    }

    let mut ans = vec![];

    while !q.is_empty() {
        for n in q.split_off(0) {
            ans.push(n as i32);
            for &in_e in in_edges[n].iter() {
                out_degree[in_e] -= 1;
            }
        }
        for (o, out_degree) in out_degree.iter_mut().enumerate() {
            if *out_degree == 0 {
                q.push_back(o);
                *out_degree = i32::MAX;
            }
        }
    }
    ans
}

fn cal_order(k: i32, conditions: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::{HashMap, VecDeque};
    let mut dep_map: HashMap<i32, Vec<i32>> = HashMap::new();

    let mut in_degree = vec![0; k as usize + 1];
    for row_cnd in conditions.iter() {
        in_degree[row_cnd[0] as usize] += 1;
        dep_map.entry(row_cnd[1]).or_default().push(row_cnd[0]);
    }
    let mut q = VecDeque::new();
    for i in 1..in_degree.len() {
        if in_degree[i] == 0 {
            q.push_back(i as i32);
            in_degree[i] = i32::MAX;
        }
    }
    let mut order = vec![];
    while let Some(n) = q.pop_front() {
        order.push(n);
        if let Some(next) = dep_map.get(&n) {
            for &n in next {
                in_degree[n as usize] -= 1;
            }
            for i in 1..in_degree.len() {
                if in_degree[i] == 0 {
                    q.push_back(i as i32);
                    in_degree[i] = i32::MAX;
                }
            }
        }
    }
    println!("{:?}", order);

    order
}
pub fn build_matrix(
    k: i32,
    row_conditions: Vec<Vec<i32>>,
    col_conditions: Vec<Vec<i32>>,
) -> Vec<Vec<i32>> {
    let row_order = cal_order(k, row_conditions);
    let col_order = cal_order(k, col_conditions);
    if row_order.is_empty() || col_order.is_empty() {
        return vec![];
    }
    let mut idx = vec![0; k as usize + 1];
    let mut ans = vec![vec![0; k as usize]; k as usize];
    for (i, n) in row_order.into_iter().enumerate() {
        ans[k as usize - i - 1][k as usize - 1] = n;
        idx[n as usize + 1] = k as usize - i - 1;
    }
    for (j, n) in col_order.into_iter().enumerate() {
        let i = idx[n as usize - 1];
        ans[i][k as usize - 1] = 0;
        ans[i][k as usize - 1 - i] = n;
    }
    ans
}

pub fn find_all_recipes(
    recipes: Vec<String>,
    ingredients: Vec<Vec<String>>,
    supplies: Vec<String>,
) -> Vec<String> {
    use std::collections::{HashMap, VecDeque};
    let mut ingredient_map: HashMap<&String, Vec<&String>> = HashMap::new();
    for (i, ingredient) in ingredients.iter().enumerate() {
        for ig in ingredient {
            ingredient_map.entry(ig).or_default().push(&recipes[i]);
        }
    }
    let mut in_deg = ingredients
        .iter()
        .enumerate()
        .map(|x| (&recipes[x.0], x.1.len()))
        .collect::<HashMap<&String, usize>>();
    for s in supplies {
        if let Some(e) = ingredient_map.get(&&s) {
            for food in e {
                in_deg.entry(food).and_modify(|v| *v -= 1);
            }
        }
    }
    let mut q = VecDeque::new();
    for (food, ideg) in in_deg.iter() {
        if *ideg == 0 {
            q.push_back(*food);
        }
    }
    let mut ans = vec![];
    while !q.is_empty() {
        for f in q.split_off(0) {
            ans.push(f.clone());
            if let Some(e) = ingredient_map.get(f) {
                for food in e {
                    in_deg.entry(food).and_modify(|v| *v -= 1);
                }
                for (food, ideg) in in_deg.iter() {
                    if *ideg == 0 {
                        q.push_back(*food);
                    }
                }
            }
        }
    }
    ans
}

pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::VecDeque;
    let mut in_degree = vec![0; num_courses as usize];
    let mut next = vec![vec![]; num_courses as usize];
    for pre in prerequisites {
        in_degree[pre[0] as usize] += 1;
        next[pre[1] as usize].push(pre[0]);
    }
    let mut q = VecDeque::new();

    let mut ans = vec![];
    for (i, &in_d) in in_degree.iter().enumerate() {
        if in_d == 0 {
            q.push_back(i);
        }
    }

    while let Some(n) = q.pop_front() {
        ans.push(n as i32);
        for &next in next[n].iter() {
            in_degree[next as usize] -= 1;
            if in_degree[next as usize] == 0 {
                q.push_back(next as usize);
            }
        }
    }

    if ans.len() != num_courses as usize {
        vec![]
    } else {
        ans
    }
}
