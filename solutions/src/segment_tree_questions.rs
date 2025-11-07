use crate::{
    segment_tree::{Mark, MyLazySegTree, MyLazySegTreeMultiMark, MyLazySumSegTree, MySegTree},
    tree_array::TreeArray3,
};

pub fn num_of_unplaced_fruits(fruits: Vec<i32>, baskets: Vec<i32>) -> i32 {
    let mut seg_tree = MySegTree::new(baskets);
    let mut res = fruits.len();
    let len = fruits.len();
    for f in fruits {
        let mut l = 1;
        let mut r = len as i32;
        while l <= r {
            let mid = (r - l) / 2 + l;
            if seg_tree.query(1, mid as usize) >= f {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if l as usize <= len {
            res -= 1;
            seg_tree.update(l as usize, -1);
        }
    }
    res as _
}

pub fn leftmost_building_queries(heights: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
    let len = heights.len();
    let mut seg_tree = MySegTree::new(heights.clone());
    queries
        .into_iter()
        .map(|q| {
            let mut l = q[0].max(q[1]) + 1;
            let mut r = len as i32;
            println!("{} {}", l, r);
            if heights[q[0].min(q[1]) as usize] < heights[q[0].max(q[1]) as usize] || q[0] == q[1] {
            } else {
                while l <= r {
                    let mid = (r - l) / 2 + l;
                    let max = seg_tree.query(l as usize, mid as usize);
                    if max > heights[q[0] as usize] && max > heights[q[1] as usize] {
                        r = mid - 1;
                    } else {
                        l = mid + 1;
                    }
                }
            }

            if (l as usize) <= len { l - 1 } else { -1 }
        })
        .collect()
}

struct BookMyShow {
    seat: MySegTree,
    col: i32,
    available_seat: TreeArray3,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl BookMyShow {
    fn new(n: i32, m: i32) -> Self {
        Self {
            seat: MySegTree::new([m].repeat(n as usize).to_vec()),
            col: m,
            available_seat: TreeArray3::new([m].repeat(n as usize).to_vec()),
        }
    }

    fn gather(&mut self, k: i32, max_row: i32) -> Vec<i32> {
        let mut l = 1;
        let mut r = max_row + 1;
        while l <= r {
            let mid = (r - l) / 2 + l;
            if self.seat.query(l as usize, mid as usize) >= k {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        if l <= max_row {
            let res = vec![l as i32 - 1, self.col - self.seat.nodes[l as usize - 1]];
            self.seat
                .update(l as usize, self.seat.nodes[l as usize - 1] - k);
            self.available_seat.update_delta(l as usize, -k);
            res
        } else {
            vec![]
        }
    }

    fn scatter(&mut self, mut k: i32, max_row: i32) -> bool {
        if self.available_seat.query_isize(1, max_row as usize + 1) >= k as isize {
            let mut i = 0;
            while k > 0 {
                let delta = self.seat.nodes[i].min(k);
                self.available_seat.update_delta(i + 1, -delta);
                self.seat.update(i + 1, self.seat.nodes[i] - delta);
                k -= delta;
                i += 1;
            }
            true
        } else {
            false
        }
    }
}

pub fn falling_squares(positions: Vec<Vec<i32>>) -> Vec<i32> {
    let len = positions.iter().map(|x| x[0] + x[1]).max().unwrap() as usize;
    let mut lazy_seg_tree = MyLazySegTree::new(vec![0; len + 1]);
    positions
        .into_iter()
        .map(|x| {
            let max = lazy_seg_tree.query_value(x[0] as usize + 1, x[0] as usize + x[1] as usize);
            lazy_seg_tree.update_range_value(
                x[0] as usize + 1,
                x[0] as usize + x[1] as usize,
                max + x[1],
            );
            lazy_seg_tree.query_value(1, len)
        })
        .collect()
}

pub fn handle_query(nums1: Vec<i32>, nums2: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i64> {
    let len = nums1.len();
    let mut current_sum = nums2.into_iter().map(|x| x as i64).sum::<i64>();
    let mut lazy_seg_tree = MyLazySumSegTree::new(nums1);
    let mut ans = vec![];
    for q in queries {
        if q[0] == 1 {
            lazy_seg_tree.update_range_value(q[1] as usize + 1, q[2] as usize + 1);
        } else if q[0] == 2 {
            current_sum += lazy_seg_tree.query_value(1, len) as i64 * q[1] as i64;
        } else {
            ans.push(current_sum);
        }
    }
    ans
}

struct Fancy {
    tree: MyLazySegTreeMultiMark,
    len: usize,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl Fancy {
    fn new() -> Self {
        Self {
            tree: MyLazySegTreeMultiMark::new(vec![0; 1_000_01]),
            len: 0,
        }
    }

    fn append(&mut self, val: i32) {
        self.len += 1;
        self.tree
            .update_range_value(self.len, self.len, Mark::Add(val as usize));
    }

    fn add_all(&mut self, inc: i32) {
        if self.len > 0 {
            self.tree
                .update_range_value(1, self.len, Mark::Add(inc as usize));
        }
    }

    fn mult_all(&mut self, m: i32) {
        if self.len > 0 {
            self.tree
                .update_range_value(1, self.len, Mark::Mul(m as usize));
        }
    }

    fn get_index(&mut self, idx: i32) -> i32 {
        if idx as usize >= self.len {
            return -1;
        } else {
            (self.tree.query_value(idx as usize + 1, idx as usize + 1) % 1_000_000_007) as i32
        }
    }
}
