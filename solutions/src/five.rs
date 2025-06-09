use std::cell::RefCell;
use std::rc::Rc;

use crate::TreeNode;
struct Solution {}
impl Solution {
    //503
    pub fn find_mode(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let mut max = 0;

        fn dfs(root: Option<Rc<RefCell<TreeNode>>>, max: &mut i32, map: &mut HashMap<i32, i32>) {
            if let Some(root) = root {
                let r = root.borrow();
                let m = map.remove(&r.val).unwrap_or(0) + 1;
                *max = (*max).max(m);
                map.insert(r.val, m);
                dfs(r.left.clone(), max, map);
                dfs(r.right.clone(), max, map);
            }
        }
        dfs(root, &mut max, &mut map);
        map.into_iter()
            .filter(|(k, v)| *v == max)
            .map(|(k, _)| k)
            .collect()
    }

    //504
    pub fn convert_to_base7(mut num: i32) -> String {
        let is_neg = num < 0;

        let mut res = String::new();
        if is_neg {
            num = -num;
        }
        while num != 0 {
            res.push_str((num % 7).to_string().as_str());
            num /= 7;
        }
        if is_neg {
            res.push('-');
        }
        unsafe {
            res.as_bytes_mut().reverse();
        }
        res
    }

    //507
    pub fn check_perfect_number(num: i32) -> bool {
        let mut sum = 0;
        let x = num as f32;
        for i in 1..=x.sqrt() as i32 {
            if num % i == 0 {
                sum += i;
                sum += num / i;
            }
        }
        sum == num
    }

    //508
    pub fn find_frequent_tree_sum(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let mut max = 0;
        fn dfs(
            root: Option<Rc<RefCell<TreeNode>>>,
            map: &mut HashMap<i32, i32>,
            max: &mut i32,
        ) -> i32 {
            if let Some(root) = root {
                let r = root.borrow();
                if r.left.is_none() && r.right.is_none() {
                    let x = map.remove(&r.val).unwrap_or(0);
                    *max = (*max).max(x + 1);
                    map.insert(r.val, x + 1);
                    return r.val;
                }

                let left = dfs(r.left.clone(), map, max);
                let right = dfs(r.right.clone(), map, max);
                let x = r.val + left + right;
                let z = map.remove(&x).unwrap_or(0);
                map.insert(x, z + 1);
                *max = (*max).max(z + 1);
                x
            } else {
                0
            }
        }
        dfs(root, &mut map, &mut max);
        map.into_iter()
            .filter(|(k, v)| *v == max)
            .map(|(k, v)| k)
            .collect()
    }

    //513
    pub fn find_bottom_left_value(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        use std::collections::VecDeque;
        let mut q = VecDeque::new();
        q.push_back(root.unwrap().clone());
        let mut res = 0;
        while !q.is_empty() {
            let mut pq = q.split_off(0);
            if let Some(r) = pq.pop_front() {
                res = r.borrow().val;
                if let Some(left) = r.borrow().left.as_ref() {
                    q.push_back(left.clone());
                }
                if let Some(right) = r.borrow().right.as_ref() {
                    q.push_back(right.clone());
                }
            }
            while let Some(r) = pq.pop_front() {
                if let Some(left) = r.borrow().left.as_ref() {
                    q.push_back(left.clone());
                }
                if let Some(right) = r.borrow().right.as_ref() {
                    q.push_back(right.clone());
                }
            }
        }

        res
    }
    fn solution_519() {
        use rand::Rng;
        use rand::rngs::ThreadRng;
        use std::collections::HashSet;
        struct Solution {
            r: ThreadRng,
            set: HashSet<(i32, i32)>,
            deleted: HashSet<(i32, i32)>,
            m: i32,
            n: i32,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl Solution {
            fn new(m: i32, n: i32) -> Self {
                let r = rand::thread_rng();
                let set = HashSet::with_capacity((n * m) as usize);
                let deleted = HashSet::with_capacity((n * m) as usize);
                let mut x = Self {
                    r,
                    set,
                    m,
                    n,
                    deleted,
                };
                for i in 0..x.m {
                    for j in 0..x.n {
                        x.set.insert((i, j));
                    }
                }
                x
            }

            fn flip(&mut self) -> Vec<i32> {
                let x = self.set.iter().map(|x| *x).collect::<Vec<(i32, i32)>>();
                println!("{:?}", x);
                let idx = self.r.gen_range(0..x.len());
                self.set.remove(&x[idx]);
                self.deleted.insert(x[idx]);
                vec![x[idx].0, x[idx].1]
            }

            fn reset(&mut self) {
                for i in self.deleted.iter() {
                    self.set.insert(*i);
                }
                self.deleted.clear();
            }
        }
    }

    //1052
    pub fn max_satisfied(customers: Vec<i32>, grumpy: Vec<i32>, minutes: i32) -> i32 {
        let mut sum = customers
            .iter()
            .zip(grumpy.iter())
            .filter(|(x, y)| **y == 0)
            .map(|(x, y)| x)
            .sum::<i32>();
        println!("{:?}", sum);
        let mut res = sum;

        let mut p = 0;

        for i in 0..minutes as usize {
            if grumpy[i] == 1 {
                p += customers[i];
            }
        }
        for i in minutes as usize..=customers.len() - minutes as usize {
            let mut x = p;
            if grumpy[i + minutes as usize] == 1 {
                x += customers[i + minutes as usize];
            }
            if grumpy[i - i] == 1 {
                x -= customers[i + minutes as usize];
            }
            p = p.max(x);
        }
        res + p
    }
}
