pub fn crack_safe(n: i32, k: i32) -> String {
    let mut map = vec![vec![]; n as usize];
    for i in 0..k as usize {
        for j in 0..k as usize {
            map[i].push(j);
        }
    }
    let mut res = String::new();
    let current = 0;
    dfs_crack_safe(current, &mut map, &mut res);
    res
}
fn dfs_crack_safe(current:usize,map:&mut Vec<Vec<usize>>,res:&mut String) {
    res.push_str(&current.to_string());
    while let Some(next) = map[current].pop() {
        dfs_crack_safe(current, map,res);
    }
}
pub fn find_itinerary(tickets: Vec<Vec<String>>) -> Vec<String> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};
    let mut map: HashMap<&str, BinaryHeap<Reverse<&str>>> = HashMap::new();
    for t in tickets.iter() {
        map.entry(&t[0]).or_default().push(Reverse(&t[1]));
    }

    let mut ans = vec![];
    dfs_jfk("JFK", &mut map, &mut ans);
    ans.reverse();
    ans
}
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::mpsc;
fn dfs_jfk(
    current: &str,
    map: &mut HashMap<&str, BinaryHeap<Reverse<&str>>>,
    stack: &mut Vec<String>,
) {
    while let Some(Reverse(next)) = map.get_mut(current).and_then(|pq| pq.pop()) {
        dfs_jfk(next, map, stack);
    }
    stack.push(current.to_string());
}
