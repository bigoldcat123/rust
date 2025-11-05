struct SegTree {
    seg_node: Vec<i32>,
    baskets: Vec<i32>,
}

impl SegTree {
    fn new(baskets: Vec<i32>) -> Self {
        let n = baskets.len();
        let mut tree = SegTree {
            seg_node: vec![0; 4 * n + 7],
            baskets,
        };
        tree.build(1, 0, n - 1);
        tree
    }

    fn build(&mut self, p: usize, l: usize, r: usize) {
        if l == r {
            self.seg_node[p] = self.baskets[l];
            return;
        }
        let mid = (l + r) / 2;
        self.build(p * 2, l, mid);
        self.build(p * 2 + 1, mid + 1, r);
        self.seg_node[p] = self.seg_node[p * 2].max(self.seg_node[p * 2 + 1]);
    }

    fn query(&self, p: usize, l: usize, r: usize, ql: usize, qr: usize) -> i32 {
        if ql > r || qr < l {
            return i32::MIN;
        }
        if ql <= l && r <= qr {
            return self.seg_node[p];
        }
        let mid = (l + r) / 2;
        self.query(p * 2, l, mid, ql, qr)
            .max(self.query(p * 2 + 1, mid + 1, r, ql, qr))
    }

    fn update(&mut self, p: usize, l: usize, r: usize, pos: usize, val: i32) {
        if l == r {
            self.baskets[pos] = val;
            self.seg_node[p] = val;
            return;
        }
        let mid = (l + r) / 2;
        if pos <= mid {
            self.update(p * 2, l, mid, pos, val);
        } else {
            self.update(p * 2 + 1, mid + 1, r, pos, val);
        }
        self.seg_node[p] = self.seg_node[p * 2].max(self.seg_node[p * 2 + 1]);
    }
}

pub struct MySegTree {
    pub  nodes:Vec<i32>,
    tree:Vec<i32>
}
impl MySegTree {
    pub fn new(nodes:Vec<i32>) -> Self {
        let len = nodes.len();
        let mut tree = vec![0;len * 4 + 7];
        let mut seg_tree = Self { nodes, tree};
        seg_tree.build(1, len, 1);
        seg_tree
    }
    fn build(&mut self,l:usize,r:usize,idx:usize) {
        if l == r {
            self.tree[idx] = self.nodes[l - 1];
        }else {
            let mid = (r + l) / 2;
            self.build(l, mid, idx * 2);
            self.build(mid + 1, r, idx * 2 + 1);
            self.tree[idx] = self.tree[idx * 2].max(self.tree[idx * 2 + 1]);
        }
    }
    /// 1-indexed
    pub fn update_delta(&mut self,idx:usize,delta:i32) {
        let value = self.nodes[idx - 1]  + delta;
        self.update(idx,value);
    }
    /// 1-indexed
    pub fn update(&mut self,idx:usize,value:i32) {
        self.nodes[idx - 1] = value;
        self.update_dfs(idx, 1, self.nodes.len(),value,1);
    }
    fn update_dfs(&mut self,idx:usize,l:usize,r:usize,value:i32,tree_idx:usize) {
        if l == r {
            self.tree[tree_idx] = value;
        }else {
            let mid = (l + r) / 2;
            if idx <= mid {
                self.update_dfs(idx, l, mid, value,tree_idx * 2);
            }else {
                self.update_dfs(idx, mid + 1, r, value,tree_idx * 2 + 1);
            }
            self.tree[tree_idx] = self.tree[tree_idx * 2].max(self.tree[tree_idx * 2 + 1]);
        }
    }
    fn query_dfs(&self,ql:usize,qr:usize,l:usize,r:usize,tree_idx:usize) -> i32 {
        if ql == l && qr == r {
            return self.tree[tree_idx];
        }
        let mid = (l + r) / 2;
        if ql > mid {
            return self.query_dfs(ql, qr, mid + 1, r, tree_idx * 2 + 1);
        }
        if qr <= mid {
            return self.query_dfs(ql, qr, l, mid, tree_idx * 2);
        }
        self.query_dfs(ql, mid, l, mid, tree_idx * 2)
            .max(self.query_dfs(mid + 1, qr, mid + 1, r, tree_idx * 2 + 1))
    }
    /// 1-indexed
    pub fn query(&self,ql:usize,qr:usize) -> i32 {
        self.query_dfs(ql, qr, 1, self.nodes.len(), 1)
    }
}
pub struct MyLazySegTree {
    pub  nodes:Vec<i32>,
    tree:Vec<(i32,i32)>// value and lazy mark
}
impl MyLazySegTree {
    pub fn new(nodes:Vec<i32>) -> Self {
        let len = nodes.len();
        let mut tree = vec![(0,0);len * 4 + 7];
        let mut seg_tree = Self { nodes, tree};
        seg_tree.build(1, len, 1);
        seg_tree
    }
    fn build(&mut self,l:usize,r:usize,idx:usize) {
        if l == r {
            self.tree[idx] = (self.nodes[l - 1],0);
        }else {
            let mid = (r + l) / 2;
            self.build(l, mid, idx * 2);
            self.build(mid + 1, r, idx * 2 + 1);
            self.tree[idx] = self.tree[idx * 2].max(self.tree[idx * 2 + 1]);
        }
    }

    pub fn update_range_delta(&mut self,range_l:usize,range_r:usize,delta:i32) {
        self.update_range_dfs(range_l, range_r, 1, self.nodes.len(), 1, delta);
    }
    fn update_range_dfs(&mut self,range_l:usize,range_r:usize,l:usize,r:usize,tree_idx:usize,delta:i32) {
        if self.tree[tree_idx].1 != 0 && l != r {
            // println!("{} {} {}",l,r,tree_idx);
            self.handle_lazy_mark(tree_idx);
        }
        if range_l == l && range_r == r {
            self.tree[tree_idx].0 += delta;
            self.tree[tree_idx].1 += delta;
        }else {
            let mid = (r + l) / 2;
            if mid >= range_r{// left
                self.update_range_dfs(range_l, range_r, l, mid, tree_idx * 2, delta);
            }else if mid < range_l {// right
                self.update_range_dfs(range_l, range_r, mid + 1, r, tree_idx * 2 + 1, delta);
            }else {
                self.update_range_dfs(range_l, mid, l, mid, tree_idx * 2, delta);
                self.update_range_dfs(mid + 1, range_r, mid + 1, r, tree_idx * 2 + 1, delta);
            }
            self.tree[tree_idx].0 = self.tree[tree_idx * 2].0.max(self.tree[tree_idx * 2 + 1].0);

        }
    }
    pub fn update_range_value(&mut self,range_l:usize,range_r:usize,value:i32) {
        self.update_range_value_dfs(range_l, range_r, 1, self.nodes.len(), 1, value);
    }
    fn update_range_value_dfs(&mut self,range_l:usize,range_r:usize,l:usize,r:usize,tree_idx:usize,value:i32) {
        if self.tree[tree_idx].1 != 0 && l != r {
            // println!("{} {} {}",l,r,tree_idx);
            self.handle_lazy_mark_value(tree_idx);
        }
        if range_l == l && range_r == r {
            self.tree[tree_idx].0 = value;
            self.tree[tree_idx].1 = value;
        }else {
            let mid = (r + l) / 2;
            if mid >= range_r{// left
                self.update_range_value_dfs(range_l, range_r, l, mid, tree_idx * 2, value);
            }else if mid < range_l {// right
                self.update_range_value_dfs(range_l, range_r, mid + 1, r, tree_idx * 2 + 1, value);
            }else {
                self.update_range_value_dfs(range_l, mid, l, mid, tree_idx * 2, value);
                self.update_range_value_dfs(mid + 1, range_r, mid + 1, r, tree_idx * 2 + 1, value);
            }
            self.tree[tree_idx].0 = self.tree[tree_idx * 2].0.max(self.tree[tree_idx * 2 + 1].0);

        }
    }
    fn handle_lazy_mark(&mut self,idx:usize) {
        let delta = self.tree[idx].1;
        self.tree[idx].1 = 0;
        self.tree[idx * 2].0 += delta;
        self.tree[idx * 2].1 += delta;
        self.tree[idx * 2 + 1].0 += delta;
        self.tree[idx * 2 + 1].1 += delta;
    }
    fn handle_lazy_mark_value(&mut self,idx:usize) {
        let delta = self.tree[idx].1;
        self.tree[idx].1 = 0;
        self.tree[idx * 2].0 = delta;
        self.tree[idx * 2].1 = delta;
        self.tree[idx * 2 + 1].0 = delta;
        self.tree[idx * 2 + 1].1 = delta;
    }
    fn query_dfs(&mut self,ql:usize,qr:usize,l:usize,r:usize,tree_idx:usize) -> i32 {
        if self.tree[tree_idx].1 != 0 && l != r {
            self.handle_lazy_mark(tree_idx);
        }
        if ql == l && qr == r {
            return self.tree[tree_idx].0;
        }
        let mid = (l + r) / 2;

        if ql > mid {
            return self.query_dfs(ql, qr, mid + 1, r, tree_idx * 2 + 1);
        }
        if qr <= mid {
            return self.query_dfs(ql, qr, l, mid, tree_idx * 2);
        }
        self.query_dfs(ql, mid, l, mid, tree_idx * 2)
            .max(self.query_dfs(mid + 1, qr, mid + 1, r, tree_idx * 2 + 1))
    }
    /// 1-indexed
    pub fn query(&mut self,ql:usize,qr:usize) -> i32 {
        self.query_dfs(ql, qr, 1, self.nodes.len(), 1)
    }
    fn query_dfs_value(&mut self,ql:usize,qr:usize,l:usize,r:usize,tree_idx:usize) -> i32 {
        if self.tree[tree_idx].1 != 0 && l != r {
            self.handle_lazy_mark_value(tree_idx);
        }
        if ql == l && qr == r {
            return self.tree[tree_idx].0;
        }
        let mid = (l + r) / 2;

        if ql > mid {
            return self.query_dfs_value(ql, qr, mid + 1, r, tree_idx * 2 + 1);
        }
        if qr <= mid {
            return self.query_dfs_value(ql, qr, l, mid, tree_idx * 2);
        }
        self.query_dfs_value(ql, mid, l, mid, tree_idx * 2)
            .max(self.query_dfs_value(mid + 1, qr, mid + 1, r, tree_idx * 2 + 1))
    }
    /// 1-indexed
    pub fn query_value(&mut self,ql:usize,qr:usize) -> i32 {
        self.query_dfs_value(ql, qr, 1, self.nodes.len(), 1)
    }
}



pub struct MyLazySumSegTree {
    pub  nodes:Vec<i32>,
    tree:Vec<(i32,i32)>// value and lazy mark
}
impl MyLazySumSegTree {
    pub fn new(nodes:Vec<i32>) -> Self {
        let len = nodes.len();
        let mut tree = vec![(0,0);len * 4 + 7];
        let mut seg_tree = Self { nodes, tree};
        seg_tree.build(1, len, 1);
        seg_tree
    }
    fn build(&mut self,l:usize,r:usize,idx:usize) {
        if l == r {
            self.tree[idx] = (self.nodes[l - 1],0);
        }else {
            let mid = (r + l) / 2;
            self.build(l, mid, idx * 2);
            self.build(mid + 1, r, idx * 2 + 1);
            self.tree[idx].0 = self.tree[idx * 2].0 + (self.tree[idx * 2 + 1]).0;
        }
    }


    pub fn update_range_value(&mut self,range_l:usize,range_r:usize) {
        self.update_range_value_dfs(range_l, range_r, 1, self.nodes.len(), 1);
    }
    fn update_range_value_dfs(&mut self,range_l:usize,range_r:usize,l:usize,r:usize,tree_idx:usize) {
        if self.tree[tree_idx].1 != 0 && l != r {
            // println!("{} {} {}",l,r,tree_idx);
            self.handle_lazy_mark_value(tree_idx,l,r);
        }
        if range_l == l && range_r == r {
            let range_len = range_r - range_l + 1;
            let value = range_len as i32 - self.tree[tree_idx].0;
            self.tree[tree_idx].0 = value;
            self.tree[tree_idx].1 = (self.tree[tree_idx].1 + 1) % 2;
        }else {
            let mid = (r + l) / 2;
            if mid >= range_r{// left
                self.update_range_value_dfs(range_l, range_r, l, mid, tree_idx * 2);
            }else if mid < range_l {// right
                self.update_range_value_dfs(range_l, range_r, mid + 1, r, tree_idx * 2 + 1);
            }else {
                self.update_range_value_dfs(range_l, mid, l, mid, tree_idx * 2);
                self.update_range_value_dfs(mid + 1, range_r, mid + 1, r, tree_idx * 2 + 1);
            }
            self.tree[tree_idx].0 = self.tree[tree_idx * 2].0 + (self.tree[tree_idx * 2 + 1].0);
        }
    }
    fn handle_lazy_mark_value(&mut self,idx:usize,l:usize,r:usize) {
        self.tree[idx].1 = 0;
        let mid = (l + r) / 2;
        let range_len_left = mid - l + 1;
        let value = range_len_left as i32 - self.tree[idx * 2].0;
        self.tree[idx * 2].0 = value;
        self.tree[idx * 2].1 = (self.tree[idx * 2].1 + 1) % 2;
        let range_len_right = r - (mid + 1) + 1;
        let value = range_len_right as i32 - self.tree[idx * 2 + 1].0;
        self.tree[idx * 2 + 1].0 = value;
        self.tree[idx * 2 + 1].1 = (self.tree[idx * 2 + 1].1 + 1) % 2;

    }

    fn query_dfs_value(&mut self,ql:usize,qr:usize,l:usize,r:usize,tree_idx:usize) -> i32 {
        if self.tree[tree_idx].1 != 0 && l != r {
            self.handle_lazy_mark_value(tree_idx,l,r);
        }
        if ql == l && qr == r {
            return self.tree[tree_idx].0;
        }
        let mid = (l + r) / 2;

        if ql > mid {
            return self.query_dfs_value(ql, qr, mid + 1, r, tree_idx * 2 + 1);
        }
        if qr <= mid {
            return self.query_dfs_value(ql, qr, l, mid, tree_idx * 2);
        }
        self.query_dfs_value(ql, mid, l, mid, tree_idx * 2)
        + (self.query_dfs_value(mid + 1, qr, mid + 1, r, tree_idx * 2 + 1))
    }
    /// 1-indexed
    pub fn query_value(&mut self,ql:usize,qr:usize) -> i32 {
        self.query_dfs_value(ql, qr, 1, self.nodes.len(), 1)
    }
}
#[cfg(test)]
mod test {
    use crate::segment_tree::MyLazySegTree;

    use super::{MySegTree, SegTree};

    /// Basic consistency test across implementations for a small fixed array.
    #[test]
    fn test_query_basic_consistency() {
        let arr = vec![1, 3, 2, 7, 5];
        let seg = SegTree::new(arr.clone());
        let my = MySegTree::new(arr.clone());
        let mut lazy = MyLazySegTree::new(arr.clone());

        let n = arr.len();
        for i in 0..n {
            for j in i..n {
                let expected = seg.query(1, 0, n - 1, i, j);
                let got_my = my.query(i + 1, j + 1);
                let got_lazy = lazy.query_value(i + 1, j + 1);
                assert_eq!(expected, got_my, "MySegTree mismatch for range {}..={}", i, j);
                assert_eq!(expected, got_lazy, "MyLazySegTree mismatch for range {}..={}", i, j);
            }
        }
    }

    /// Query single elements should return the element itself.
    #[test]
    fn test_query_single_element() {
        let arr = vec![10, -5, 0, 42];
        let seg = SegTree::new(arr.clone());
        let my = MySegTree::new(arr.clone());
        let mut lazy = MyLazySegTree::new(arr.clone());

        for idx in 0..arr.len() {
            let expected = seg.query(1, 0, arr.len() - 1, idx, idx);
            let got_my = my.query(idx + 1, idx + 1);
            let got_lazy = lazy.query_value(idx + 1, idx + 1);
            assert_eq!(expected, got_my);
            assert_eq!(expected, got_lazy);
        }
    }

    /// Query full range should return the global maximum.
    #[test]
    fn test_query_full_range() {
        let arr = vec![2, 9, 1, 8, 7, 3];
        let seg = SegTree::new(arr.clone());
        let my = MySegTree::new(arr.clone());
        let mut lazy = MyLazySegTree::new(arr.clone());

        let expected = seg.query(1, 0, arr.len() - 1, 0, arr.len() - 1);
        let got_my = my.query(1, arr.len());
        let got_lazy = lazy.query_value(1, arr.len());
        assert_eq!(expected, got_my);
        assert_eq!(expected, got_lazy);
    }

    /// Ensure queries remain correct after point and range updates.
    #[test]
    fn test_query_after_updates() {
        let mut arr = vec![5, 4, 3, 2, 1];
        let mut seg = SegTree::new(arr.clone());
        let mut my = MySegTree::new(arr.clone());
        let mut lazy = MyLazySegTree::new(arr.clone());

        // point update: set index 2 (0-based) -> 100
        seg.update(1, 0, arr.len() - 1, 2, 100);
        my.update(3, 100); // 1-indexed
        lazy.update_range_value(3, 3, 100);

        // after update, maximum over full range should reflect the new value
        let expected = seg.query(1, 0, arr.len() - 1, 0, arr.len() - 1);
        assert_eq!(expected, my.query(1, arr.len()));
        assert_eq!(expected, lazy.query_value(1, arr.len()));

        // range update via lazy: add 10 to positions 2..4 (1-indexed)
        lazy.update_range_delta(2, 4, 10);
        // apply same increments to my and seg by updating points individually for comparison
        for pos in 2..=4 {
            // seg is 0-indexed
            let current = seg.query(1, 0, arr.len() - 1, pos - 1, pos - 1);
            seg.update(1, 0, arr.len() - 1, pos - 1, current + 10);
            my.update(pos, my.query(pos, pos) + 10);
        }

        // now compare all subrange queries
        for i in 0..arr.len() {
            for j in i..arr.len() {
                let e = seg.query(1, 0, arr.len() - 1, i, j);
                let m = my.query(i + 1, j + 1);
                let l = lazy.query_value(i + 1, j + 1);
                assert_eq!(e, m, "mismatch after updates for range {}..={}", i, j);
                assert_eq!(e, l, "lazy mismatch after updates for range {}..={}", i, j);
            }
        }
    }
}
#[derive(Debug,Clone,Copy)]
pub enum Mark {
    Add(usize),
    Mul(usize)
}
#[derive(Debug,Clone,Default)]
struct TreeNode {
    v:usize,
    marks:Vec<Mark>
}
impl TreeNode {
    fn new(v:usize,marks:Vec<Mark>) -> Self {
        TreeNode { v, marks }
    }
}
pub struct MyLazySegTreeMultiMark {
    pub  nodes:Vec<usize>,
    tree:Vec<TreeNode>// value and lazy mark
}
impl MyLazySegTreeMultiMark {
    pub fn new(nodes:Vec<usize>) -> Self {
        let len = nodes.len();
        let mut tree = vec![TreeNode::default();len * 4 + 7];
        let mut seg_tree = Self { nodes, tree};
        seg_tree.build(1, len, 1);
        seg_tree
    }
    fn build(&mut self,l:usize,r:usize,idx:usize) {
        if l == r {
            self.tree[idx] = TreeNode::new(self.nodes[l - 1],vec![]);
        }else {
            let mid = (r + l) / 2;
            self.build(l, mid, idx * 2);
            self.build(mid + 1, r, idx * 2 + 1);
            self.tree[idx].v = self.tree[idx * 2].v + (self.tree[idx * 2 + 1]).v;
        }
    }
    pub fn update_range_value(&mut self,range_l:usize,range_r:usize,update_type:Mark) {
        self.update_range_value_dfs(range_l, range_r, 1, self.nodes.len(), 1, update_type);
    }
    fn update_range_value_dfs(&mut self,range_l:usize,range_r:usize,l:usize,r:usize,tree_idx:usize,update_type:Mark) {
        if !self.tree[tree_idx].marks.is_empty() && l != r {
            // println!("{} {} {}",l,r,tree_idx);
            self.handle_lazy_mark_value(tree_idx,l,r);
        }
        if range_l == l && range_r == r {
            let len = (r - l + 1);
            match update_type {
                Mark::Add(add_f) => {
                    self.tree[tree_idx].v = (self.tree[tree_idx].v + len * add_f) % 1_000_000_007;
                }
                Mark::Mul(mul_f) => {
                    self.tree[tree_idx].v = (self.tree[tree_idx].v * mul_f)% 1_000_000_007;
                }
            }
            if l != r {
                self.tree[tree_idx].marks.push(update_type);
            }
        }else {
            let mid = (r + l) / 2;
            if mid >= range_r{// left
                self.update_range_value_dfs(range_l, range_r, l, mid, tree_idx * 2, update_type);
            }else if mid < range_l {// right
                self.update_range_value_dfs(range_l, range_r, mid + 1, r, tree_idx * 2 + 1, update_type);
            }else {
                self.update_range_value_dfs(range_l, mid, l, mid, tree_idx * 2, update_type);
                self.update_range_value_dfs(mid + 1, range_r, mid + 1, r, tree_idx * 2 + 1, update_type);
            }
            self.tree[tree_idx].v = self.tree[tree_idx * 2].v + (self.tree[tree_idx * 2 + 1].v);

        }
    }

    fn handle_lazy_mark_value(&mut self,idx:usize,l:usize,r:usize) {
        let mid = (l + r) / 2;
        let len_left = (mid - l) + 1;
        let len_right = (r - (mid + 1)) + 1;
        for i in 0..self.tree[idx].marks.len() {
            match self.tree[idx].marks[i] {
                Mark::Add(add_f) => {
                    //left
                    self.tree[idx * 2].v = (self.tree[idx * 2].v + len_left * add_f) % 1_000_000_007;
                    if l != mid {
                        self.tree[idx * 2].marks.push(Mark::Add(add_f));
                    }
                    //right
                    self.tree[idx * 2 + 1].v = (self.tree[idx * 2 + 1].v + len_right * add_f) % 1_000_000_007;
                    if mid + 1 !=r {
                        self.tree[idx * 2 + 1].marks.push(Mark::Add(add_f));
                    }
                }
                Mark::Mul(mul_f) => {
                    //left
                    self.tree[idx * 2].v = (self.tree[idx * 2].v * mul_f)% 1_000_000_007;
                    if l != mid {
                        self.tree[idx * 2].marks.push(Mark::Mul(mul_f));
                    }
                    //right
                    self.tree[idx * 2 + 1].v = (self.tree[idx * 2 + 1].v * mul_f)% 1_000_000_007;
                    if mid + 1 != r {
                        self.tree[idx * 2 + 1].marks.push(Mark::Mul(mul_f));
                    }
                }
            }
        }
        self.tree[idx].marks.clear();
    }

    fn query_dfs_value(&mut self,ql:usize,qr:usize,l:usize,r:usize,tree_idx:usize) -> usize {
        if !self.tree[tree_idx].marks.is_empty() && l != r {
            self.handle_lazy_mark_value(tree_idx,l,r);
        }
        if ql == l && qr == r {
            return self.tree[tree_idx].v;
        }
        let mid = (l + r) / 2;

        if ql > mid {
            return self.query_dfs_value(ql, qr, mid + 1, r, tree_idx * 2 + 1);
        }
        if qr <= mid {
            return self.query_dfs_value(ql, qr, l, mid, tree_idx * 2);
        }
        self.query_dfs_value(ql, mid, l, mid, tree_idx * 2)
            .max(self.query_dfs_value(mid + 1, qr, mid + 1, r, tree_idx * 2 + 1))
    }
    /// 1-indexed
    pub fn query_value(&mut self,ql:usize,qr:usize) -> usize {
        self.query_dfs_value(ql, qr, 1, self.nodes.len(), 1)
    }
}
