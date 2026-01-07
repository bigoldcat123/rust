
pub fn find_itinerary(tickets: Vec<Vec<String>>) -> Vec<String> {
    use std::collections::{BinaryHeap,HashMap};
    use std::cmp::Reverse;
    let mut map:HashMap<&str,BinaryHeap<Reverse<&str>>> = HashMap::new();
    for t in tickets.iter() {
        map.entry(&t[0]).or_default().push(Reverse(&t[1]));
    }

    let mut ans = vec![];;
    dfs_jfk("JFK", &mut map, &mut ans);
    ans.reverse();
    ans
}
use std::collections::{BinaryHeap,HashMap};
use std::cmp::Reverse;
fn dfs_jfk(current:&str,map:&mut HashMap<&str,BinaryHeap<Reverse<&str>>>,stack:&mut Vec<String>) {
    while let Some(Reverse(next)) = map.get_mut(current).and_then(|pq| pq.pop()) {
        dfs_jfk(next,map,stack);
    }
    stack.push(current.to_string());
}
