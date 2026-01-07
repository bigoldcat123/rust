
struct DSet {
    fa: Vec<usize>,
    cn: usize,
}
impl DSet {
    fn new(n: usize) -> Self {
        let mut fa = (0..n).into_iter().collect();
        Self { fa, cn: n }
    }
    fn find(&mut self, n: usize) -> usize {
        if self.fa[n] == n {
            return n;
        } else {
            self.fa[n] = self.find(self.fa[n]);
        }
        self.fa[n]
    }
    fn merge(&mut self, from: usize, to: usize) -> bool {
        let x = self.find(from);
        let y = self.find(to);
        if x == y {
            return false;
        }
        self.fa[x] = y;
        self.cn -= 1;
        true
    }
}

pub fn max_stability(n: i32, mut edges: Vec<Vec<i32>>, mut k: i32) -> i32 {
    use std::collections::BinaryHeap;
    let mut set = DSet::new(n as usize);
    edges.sort_by_key(|x| x[2]);
    let mut stability = i32::MAX;
    for e in edges.iter().filter(|x| x[3] == 1).rev() {
        if set.merge(e[0] as usize, e[1] as usize) {
            stability = stability.min(e[2]);
        }else {
            return -1;
        }

    }
    if stability == i32::MAX {
        stability = edges.last().unwrap()[2];
    }
    let mut inserted_edges = BinaryHeap::new();
    let min = stability;
    for e in edges.iter().filter(|x| x[3] == 0).rev() {
        if set.merge(e[0] as usize, e[1] as usize) {
            inserted_edges.push((-e[2],true));
        }
        if set.cn == 1 {
            break;
        }
    }
    if k > 0 {
        while let Some((value,nod_double)) = inserted_edges.pop() {
            k -= 1;
            if nod_double {
                inserted_edges.push((value * 2, false));
            }else {
                break;
            }
            if k == 0 {
                break;
            }
        }
    }
    if set.cn != 1 {
        return -1
    }
    (-inserted_edges.pop().unwrap_or((0,false)).0).min(min)
}

pub fn min_cost_connect_points(points: Vec<Vec<i32>>) -> i32 {
    let mut edges = vec![];
    for i in 0..points.len() {
        let p = (points[i][0], points[i][1]);
        for j in i + 1..points.len() {
            let q = (points[j][0], points[j][1]);
            let dist = ((p.0 - q.0).abs() + (p.1 - q.1).abs()) as i32;
            edges.push((dist, i, j));
            edges.push((dist, j, i));
        }
    }
    edges.sort_by_key(|x| x.0);
    let mut set = DSet::new(edges.len());
    let mut ans = 0;
    for &(dist, i, j) in edges.iter() {
        if set.merge(i, j) {
            ans += dist;
        }
        if set.cn == 1 {
            break;
        }
    }
    ans
}
