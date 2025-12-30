
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
    head.push((0,0));
    while let Some((min_cost,node)) = head.pop() {
        let min_cost = -min_cost;
        if min_cost > ans[node] {
            continue;
        }
        for &(next, cost) in map[node].iter() {
            if min_cost + cost < disappear[next] && min_cost + cost < ans[next]{
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
