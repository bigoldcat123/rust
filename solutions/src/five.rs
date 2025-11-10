#![allow(deprecated, non_snake_case)]
// use std::borrow::Borrow;
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::iter::Take;
use std::ops::{Deref, Index};
use std::rc::Rc;
use std::thread::sleep;

use rand::rand_core::impls;

use crate::{ListNode, TreeNode};
pub struct Solution {}
impl Solution {
    //503
    // pub fn find_mode(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    //     use std::collections::HashMap;
    //     let mut map = HashMap::new();
    //     let mut max = 0;

    //     fn dfs(root: Option<Rc<RefCell<TreeNode>>>, max: &mut i32, map: &mut HashMap<i32, i32>) {
    //         if let Some(root) = root {
    //             let r:&_ = root.borrow();
    //             let m = map.remove(&r.val).unwrap_or(0) + 1;
    //             *max = (*max).max(m);
    //             map.insert(r.val, m);
    //             dfs(r.left.clone(), max, map);
    //             dfs(r.right.clone(), max, map);
    //         }
    //     }
    //     dfs(root, &mut max, &mut map);
    //     map.into_iter()
    //         .filter(|(k, v)| *v == max)
    //         .map(|(k, _)| k)
    //         .collect()
    // }

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
                let x = self.set.iter().copied().collect::<Vec<(i32, i32)>>();
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
    fn solution_528() {
        use rand::Rng;
        use rand::rngs::ThreadRng;
        struct Solution {
            nums: Vec<(usize, i32, i32)>,
            len: usize,
            rng: ThreadRng,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl Solution {
            fn new(w: Vec<i32>) -> Self {
                let nums = w
                    .into_iter()
                    .enumerate()
                    .map(|x| (x.0, x.1, x.1))
                    .collect::<Vec<(usize, i32, i32)>>();
                let rng = rand::thread_rng();
                Self {
                    len: nums.len(),
                    nums,
                    rng,
                }
            }

            fn pick_index(&mut self) -> i32 {
                if self.len == 0 {
                    self.len = self.nums.len();
                    for i in 0..self.nums.len() {
                        self.nums[i].1 = self.nums[i].2;
                    }
                    return self.pick_index();
                }
                let idx = self.rng.gen_range(0..self.len);
                self.nums[idx].1 -= 1;
                let res = self.nums[idx].0;
                if self.nums[idx].1 == 0 {
                    self.nums.swap(idx, self.len - 1);
                    self.len -= 1;
                }
                res as i32
            }
        }
    }

    //566
    pub fn matrix_reshape(mat: Vec<Vec<i32>>, r: i32, c: i32) -> Vec<Vec<i32>> {
        let max = (mat.len() * mat[0].len()) as i32;
        let target = r * c;
        if target > max {
            return mat;
        }
        let a = mat.into_iter().flatten().collect::<Vec<i32>>();
        let mut res = vec![vec![0; c as usize]; r as usize];
        let mut idx = 0;
        for i in 0..r as usize {
            for j in 0..c as usize {
                res[i][j] = a[idx];
                idx += 1;
            }
        }
        res
    }

    pub fn get_subarray_beauty(nums: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
        use std::collections::LinkedList;
        let l = LinkedList::from_iter([1].iter());
        let mut res = vec![];
        let mut p = Vec::from_iter(nums[..k as usize].iter().copied());
        p.sort();
        res.push(p[x as usize - 1].min(0));
        let (mut l, mut r) = (0, k as usize);
        while r < nums.len() {
            let idx = Self::find_idx(&p, nums[l]);
            p[idx] = nums[r];
            p.sort();
            res.push(p[x as usize - 1].min(0));
            l += 1;
            r += 1;
        }
        res
    }
    fn find_idx(nums: &[i32], target: i32) -> usize {
        let (mut l, mut r) = (0, nums.len() - 1);
        while l <= r {
            let mid = (r + l) / 2;
            if nums[mid] > target {
                r = mid - 1;
            } else if nums[mid] < target {
                l = mid + 1;
            } else {
                return mid;
            }
        }
        0
    }
    fn adasd() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        for i in 1..0 {}
    }
    pub fn check_palindrome_formation(a: String, b: String) -> bool {
        if Self::is_palindrome(a.as_bytes()) || Self::is_palindrome(b.as_bytes()) {
            return true;
        }
        let (mut l, mut r) = (0, a.len() - 1);
        let a = a.as_bytes();
        let b = b.as_bytes();
        while l < r && a[l] == b[r] {
            println!("eeeeee1 {} {}", a[l] as char, b[r] as char);

            l += 1;
            r -= 1;
        }
        while l < r && b[l] == b[r] {
            println!("eeeeee2");

            l += 1;
            r -= 1;
        }
        if l >= r {
            println!("eeeeee");
            return true;
        }
        println!("e?");

        let (mut l, mut r) = (0, a.len() - 1);
        // let a = a.as_bytes();
        // let b = b.as_bytes();
        while l < r && b[l] == a[r] {
            println!("eeeeee3");

            l += 1;
            r -= 1;
        }
        while l < r && a[l] == a[r] {
            println!("eeeeee5");

            l += 1;
            r -= 1;
        }
        if l >= r {
            println!("ee");

            return true;
        }
        false
    }
    fn is_palindrome(s: &[u8]) -> bool {
        let (mut l, mut r) = (0, s.len() - 1);
        while l < r {
            if s[l] != s[r] {
                return false;
            }
        }
        true
    }

    pub fn max_average_ratio(classes: Vec<Vec<i32>>, mut extra_students: i32) -> f64 {
        #[derive(PartialEq, Eq, Debug)]
        struct Dataa {
            pass: i32,
            total: i32,
        }
        impl PartialOrd for Dataa {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                let o = self.pass as f64 / self.total as f64;
                let big = (self.pass + 1) as f64 / (self.total + 1) as f64;
                let o2 = other.pass as f64 / other.total as f64;
                let big2 = (other.pass + 1) as f64 / (other.total + 1) as f64;
                (big - o).partial_cmp(&(big2 - o2))
            }
        }
        impl Ord for Dataa {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }
        let mut full = classes.len();
        let mut classes: BinaryHeap<Reverse<Dataa>> = classes
            .into_iter()
            .filter(|x| x[0] != x[1])
            .map(|x| {
                Reverse(Dataa {
                    pass: x[0],
                    total: x[1],
                })
            })
            .collect();
        full -= classes.len();
        println!("{:?}", classes);
        while extra_students > 0 {
            extra_students -= 1;
            let mut e = classes.pop().unwrap().0;
            e.pass += 1;
            e.total += 1;
            classes.push(Reverse(e));
        }
        let s: f64 = classes
            .iter()
            .map(|Reverse(x)| x.pass as f64 / x.total as f64)
            .sum();
        (s + full as f64) / (classes.len() + full) as f64
    }
    fn abc() {
        use std::collections::{BTreeSet, HashMap};
        #[derive(PartialEq, Eq, Clone, Copy)]
        struct Task {
            priority: i32,
            task_id: i32,
            user_id: i32,
        };
        impl PartialOrd for Task {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(
                    other
                        .priority
                        .cmp(&self.priority)
                        .then(other.user_id.cmp(&self.user_id)),
                )
            }
        }
        impl Ord for Task {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }
        struct TaskManager {
            tasks: BTreeSet<Task>,
            id_map: HashMap<i32, Task>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl TaskManager {
            //[userId, taskId, priority]
            fn new(tasks: Vec<Vec<i32>>) -> Self {
                let mut id_map = HashMap::new();
                Self {
                    tasks: BTreeSet::from_iter(tasks.into_iter().map(|x| {
                        let task = Task {
                            user_id: x[0],
                            task_id: x[1],
                            priority: x[2],
                        };
                        id_map.insert(task.task_id, task);
                        task
                    })),
                    id_map,
                }
            }

            fn add(&mut self, user_id: i32, task_id: i32, priority: i32) {
                let task = Task {
                    user_id,
                    task_id,
                    priority,
                };
                self.tasks.insert(task);
                self.id_map.insert(task_id, task);
            }

            fn edit(&mut self, task_id: i32, new_priority: i32) {
                let mut t = self.id_map.get(&task_id).copied().unwrap();
                self.tasks.remove(&t);
                t.priority = new_priority;
                self.tasks.insert(t);
            }

            fn rmv(&mut self, task_id: i32) {
                let mut t = self.id_map.get(&task_id).copied().unwrap();
                self.tasks.remove(&t);
            }

            fn exec_top(&mut self) -> i32 {
                if let Some(t) = self.tasks.pop_first() {
                    t.user_id
                } else {
                    -1
                }
            }
        }
    }
    fn p() {
        // struct MyStr{

        // }
        // // impl Borrow<str> for MyStr {
        // //     fn borrow(&self) -> &str {
        // //         "hello"
        // //     }
        // // }
        // impl Deref for MyStr {
        //     type Target = str;
        //     fn deref(&self) -> &Self::Target {
        //         "el"
        //     }
        // }
        // let s = MyStr{};
        // // let a:&str = &s;
        // // let b:&MyStr = &s;
        // fn take_str(s:&str) {

        // }
        // take_str(&s);
    }
    pub fn halve_array(nums: Vec<i32>) -> i32 {
        #[derive(Debug)]
        struct MyFloat(f64);
        impl PartialEq for MyFloat {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl Eq for MyFloat {}
        impl PartialOrd for MyFloat {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }
        impl Ord for MyFloat {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.partial_cmp(&other.0).unwrap()
            }
        }
        let a = MyFloat(1.2);
        let b = MyFloat(1.2);
        a < b;

        let sum = nums.iter().map(|&x| x as f64).sum::<f64>();
        let target = sum / 2.0;

        let mut current = 0.0;
        let mut heap = BinaryHeap::from_iter(nums.into_iter().map(|x| MyFloat(x as f64)));
        let mut res = 0;
        while current < target {
            let x = heap.pop().unwrap();
            let y = x.0 / 2.0;
            current += y;
            heap.push(MyFloat(y));
            res += 1;
        }
        res
    }
    pub fn min_number_of_seconds(mut mountain_height: i32, worker_times: Vec<i32>) -> i64 {
        struct Worker {
            work_time: i64,
            current_cost: i64,
            initial_cost: i64,
        }
        impl Worker {
            fn new(initial_cost: i64) -> Self {
                Self {
                    work_time: 1,
                    current_cost: initial_cost,
                    initial_cost,
                }
            }
            fn increase(&mut self) {
                self.work_time += 1;
                self.current_cost += self.work_time * self.initial_cost;
            }
        }
        impl PartialEq for Worker {
            fn eq(&self, other: &Self) -> bool {
                self.current_cost == other.current_cost
            }
        }
        impl Eq for Worker {}
        impl PartialOrd for Worker {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.current_cost
                    .partial_cmp(&other.current_cost)
                    .map(|x| x.reverse())
            }
        }
        impl Ord for Worker {
            fn cmp(&self, other: &Self) -> Ordering {
                self.current_cost.cmp(&other.current_cost).reverse()
            }
        }
        let mut heap =
            BinaryHeap::from_iter(worker_times.into_iter().map(|x| Worker::new(x as i64)));
        let mut res = 0;
        while mountain_height > 0 {
            let mut worker = heap.pop().unwrap();
            mountain_height -= 1;
            res += worker.current_cost;
            worker.increase();
            heap.push(worker);
        }
        res
    }
    fn a() {
        use std::cmp::Ordering;
        struct L {
            l: Option<Box<ListNode>>,
        }
        impl PartialOrd for L {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                match (self.l.as_ref(), other.l.as_ref()) {
                    (Some(s), Some(o)) => o.val.partial_cmp(&s.val),
                    (None, None) => Some(Ordering::Equal),
                    (None, Some(_)) => Some(Ordering::Less),
                    (Some(_), None) => Some(Ordering::Greater),
                }
            }
        }
        impl Ord for L {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }
        impl PartialEq for L {
            fn eq(&self, other: &Self) -> bool {
                self.partial_cmp(other).unwrap() == Ordering::Equal
            }
        }
        impl Eq for L {}

        fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
            let mut heap: BinaryHeap<L> =
                BinaryHeap::from_iter(lists.into_iter().map(|x| L { l: x }));
            let mut res = ListNode::new(0);
            let mut res_ref = &mut res;
            while let Some(mut min) = heap.peek_mut() {
                if let Some(mut min_l) = min.l.take() {
                    let next = ListNode::new(min_l.val);
                    res_ref.next = Some(Box::new(next));
                    res_ref = res_ref.next.as_mut().unwrap();
                    min.l = min_l.next;
                } else {
                    break;
                }
            }
            res.next
        }
        fn ffff() {
            use std::collections::{BTreeSet, HashMap};
            struct FoodRatings {
                cuisines: HashMap<Rc<String>, BTreeSet<(i32, Rc<Reverse<String>>)>>,
                food_cuisine: HashMap<Rc<Reverse<String>>, (Rc<String>, i32)>,
            }

            /**
             * `&self` means the method takes an immutable reference.
             * If you need a mutable reference, change it to `&mut self` instead.
             */
            impl FoodRatings {
                fn new(
                    mut foods: Vec<String>,
                    mut cuisines: Vec<String>,
                    mut ratings: Vec<i32>,
                ) -> Self {
                    let mut c = HashMap::new();
                    let mut f_c = HashMap::new();
                    let len = foods.len();
                    for i in 0..len {
                        let food = Rc::new(Reverse(foods.pop().unwrap()));
                        let cuisine = Rc::new(cuisines.pop().unwrap());
                        let rating = ratings.pop().unwrap();
                        c.entry(cuisine.clone())
                            .or_insert(BTreeSet::new())
                            .insert((rating, food.clone()));
                        f_c.insert(food.clone(), (cuisine.clone(), rating));
                    }
                    Self {
                        cuisines: c,
                        food_cuisine: f_c,
                    }
                }

                fn change_rating(&mut self, food: String, new_rating: i32) {
                    let food = Rc::new(Reverse(food));
                    let (cuisine, rating) = self.food_cuisine.get(&food).unwrap();
                    if let Some(foods) = self.cuisines.get_mut(cuisine) {
                        foods.remove(&(*rating, food.clone()));
                        foods.insert((new_rating, food.clone()));
                        self.food_cuisine
                            .insert(food, (cuisine.clone(), new_rating));
                    }
                }

                fn highest_rated(&self, cuisine: String) -> String {
                    if let Some(foods) = self.cuisines.get(&Rc::new(cuisine)) {
                        println!("{:?}", foods);
                        let res = foods.last().unwrap().1.clone();
                        (*res).clone().0
                    } else {
                        "".into()
                    }
                }

                pub fn most_frequent_i_ds(nums: Vec<i32>, freq: Vec<i32>) -> Vec<i64> {
                    use std::collections::{BTreeSet, HashMap};
                    let mut current_freq = HashMap::new();
                    let mut freq_set: BTreeSet<(i64, i32)> = BTreeSet::new();
                    let mut res = vec![0; freq.len()];
                    for i in 0..nums.len() {
                        let pre_freq = if let Some(r) = current_freq.get(&nums[i]) {
                            *r
                        } else {
                            0
                        };
                        freq_set.remove(&(pre_freq, nums[i]));
                        *current_freq.entry(nums[i]).or_insert(0) += (freq[i] as i64);
                        freq_set.insert((pre_freq + freq[i] as i64, nums[i]));

                        let r = freq_set.last().unwrap();
                        res[i] = r.0;
                    }
                    res
                }
            }
        }
        fn stock() {
            use std::collections::{BTreeSet, HashMap};
            struct StockPrice {
                latest: (i32, i32), // time,price
                time_price: HashMap<i32, i32>,
                price_set: BTreeSet<(i32, i32)>, // price,timestamp
            }

            /**
             * `&self` means the method takes an immutable reference.
             * If you need a mutable reference, change it to `&mut self` instead.
             */
            impl StockPrice {
                fn new() -> Self {
                    Self {
                        latest: (0, 0),
                        time_price: HashMap::new(),
                        price_set: BTreeSet::new(),
                    }
                }

                fn update(&mut self, timestamp: i32, price: i32) {
                    if let Some(old_price) = self.time_price.get_mut(&timestamp) {
                        self.price_set.remove(&(*old_price, timestamp));
                        *old_price = price;
                    } else {
                        self.time_price.insert(timestamp, price);
                    }
                    self.price_set.insert((price, timestamp));
                    if timestamp >= self.latest.0 {
                        self.latest.0 = timestamp;
                        self.latest.1 = price;
                    }
                }

                fn current(&self) -> i32 {
                    self.latest.1
                }

                fn maximum(&self) -> i32 {
                    self.price_set.last().unwrap().0
                }

                fn minimum(&self) -> i32 {
                    self.price_set.first().unwrap().0
                }
            }
        }
        fn infinite_stack() {
            use std::collections::BTreeSet;
            struct DinnerPlates {
                stacks: Vec<Vec<i32>>,
                len: usize,
                full: BTreeSet<usize>,
                avaliable: BTreeSet<usize>,
            }

            /**
             * `&self` means the method takes an immutable reference.
             * If you need a mutable reference, change it to `&mut self` instead.
             */
            impl DinnerPlates {
                fn new(capacity: i32) -> Self {
                    Self {
                        stacks: vec![vec![]],
                        len: capacity as usize,
                        full: BTreeSet::new(),
                        avaliable: BTreeSet::from([0]),
                    }
                }

                fn push(&mut self, val: i32) {
                    if let Some(min) = self.avaliable.pop_first() {
                        self.stacks[min].push(val);
                        if self.stacks[min].len() < self.len {
                            self.avaliable.insert(min);
                        } else {
                            self.full.insert(min);
                            if self.avaliable.is_empty() {
                                self.stacks.push(vec![]);
                                self.avaliable.insert(self.stacks.len() - 1);
                            }
                        }
                    }
                }

                fn pop(&mut self) -> i32 {
                    // for i in (0..self.stacks.len()).rev() {
                    //     if !self.stacks[i].is_empty() {
                    //         if self.stacks[i].len() == self.len {
                    //             self.full.remove((&i));
                    //             self.avaliable.insert(i);
                    //         }
                    //         return self.stacks[i].pop().unwrap()
                    //     }
                    // }
                    // -1
                    while let Some(last) = self.stacks.last_mut() {
                        if last.is_empty() {
                            if self.stacks.len() > 1 {
                                self.stacks.pop();
                                self.avaliable.remove(&self.stacks.len());
                            } else {
                                return -1;
                            }
                        } else if last.len() == self.len {
                            let res = last.pop().unwrap();
                            self.full.remove(&(self.stacks.len() - 1));
                            self.avaliable.insert((self.stacks.len() - 1));

                            return res;
                        } else {
                            let res = last.pop().unwrap();
                            return res;
                        }
                    }
                    -1
                }

                fn pop_at_stack(&mut self, index: i32) -> i32 {
                    let index = index as usize;
                    if self.stacks.len() > index {
                        if self.stacks[index].len() == self.len {
                            let res = self.stacks[index].pop().unwrap();
                            self.full.remove(&index);
                            self.avaliable.insert(index);
                            res
                        } else {
                            self.stacks[index].pop().unwrap_or(-1)
                        }
                    } else {
                        -1
                    }
                }
            }
        }
        fn sort_racker_impl() {
            use std::cmp::Reverse;
            use std::collections::BinaryHeap;
            #[derive(Default)]
            struct SORTracker {
                min_heap: BinaryHeap<Reverse<(i32, Reverse<String>)>>,
                max_heap: BinaryHeap<(i32, Reverse<String>)>,
                size: i32,
            }
            // 9 8 7 -> 9
            // 8 7   -> 8 9
            // 8 7   -> 9 9

            /**
             * `&self` means the method takes an immutable reference.
             * If you need a mutable reference, change it to `&mut self` instead.
             */
            impl SORTracker {
                fn new() -> Self {
                    Default::default()
                }

                fn add(&mut self, name: String, score: i32) {
                    // self.max_heap.push((score,name));
                    if let Some(Reverse(min)) = self.min_heap.peek() {
                        if min < &(score, Reverse(name.clone())) {
                            let x = self.min_heap.pop().unwrap().0;
                            self.max_heap.push((x.0, x.1));
                            self.min_heap.push(Reverse((score, Reverse(name))));
                        } else {
                            self.max_heap.push((score, Reverse(name)));
                        }
                    } else {
                        self.max_heap.push((score, Reverse(name)));
                    }
                }

                fn get(&mut self) -> String {
                    if let Some(res) = self.max_heap.pop() {
                        self.min_heap.push(Reverse((res.0, res.1.clone())));
                        res.1.0
                    } else {
                        "".into()
                    }
                }
            }
        }
        fn median_finder_impl() {
            use std::cmp::Reverse;
            use std::collections::BinaryHeap;
            #[derive(Default)]
            struct MedianFinder {
                small_max_heap: BinaryHeap<i32>,
                big_min_heap: BinaryHeap<Reverse<i32>>,
            }
            /**
             * `&self` means the method takes an immutable reference.
             * If you need a mutable reference, change it to `&mut self` instead.
             */
            impl MedianFinder {
                fn new() -> Self {
                    Default::default()
                }

                fn add_num(&mut self, num: i32) {
                    if self.small_max_heap.len() == self.big_min_heap.len() {
                        self.small_max_heap.push(num);
                    } else {
                        self.big_min_heap.push(Reverse(num));
                    }
                    let max_small = self.small_max_heap.peek().copied().unwrap();
                    if let Some(&Reverse(min_big)) = self.big_min_heap.peek() {
                        if max_small > min_big {
                            self.small_max_heap.pop();
                            self.big_min_heap.pop();
                            self.small_max_heap.push(min_big);
                            self.big_min_heap.push(Reverse(max_small));
                        }
                    }
                }

                fn find_median(&self) -> f64 {
                    if (self.big_min_heap.len() + self.small_max_heap.len()) % 2 == 0 {
                        (self.small_max_heap.peek().copied().unwrap()
                            + self.big_min_heap.peek().copied().unwrap().0)
                            as f64
                            / 2.0
                    } else {
                        self.small_max_heap.peek().copied().unwrap() as _
                    }
                }
            }
        }
        pub fn median_sliding_window(nums: Vec<i32>, k: i32) -> Vec<f64> {
            use std::collections::BTreeSet;
            let k = k as usize;
            let mut small_max_set = BTreeSet::new();
            let mut big_min_set = BTreeSet::new();
            for i in 0..k {
                if small_max_set.len() == big_min_set.len() + 1 {
                    big_min_set.insert((nums[i], i));
                } else {
                    small_max_set.insert((nums[i], i));
                }
                let small_max = small_max_set.last().copied().unwrap();
                if let Some(big_min) = big_min_set.first().copied() {
                    if small_max > big_min {
                        small_max_set.pop_last();
                        big_min_set.pop_first();
                        small_max_set.insert(big_min);
                        big_min_set.insert(small_max);
                    }
                }
            }
            let mut res = vec![0.0; nums.len() - k + 1];
            res[0] = if k % 2 == 0 {
                (small_max_set.last().copied().unwrap().0 as f64
                    + (big_min_set.first().copied().unwrap().0) as f64)
                    / 2.0
            } else {
                small_max_set.last().copied().unwrap().0 as f64
            };
            println!("-> {:?}", small_max_set);
            println!("{:?}", big_min_set);

            let mut l = 0;
            let mut r = k;
            while r < nums.len() {
                let del = (nums[l], l);
                let add = (nums[r], r);
                big_min_set.remove(&del);
                small_max_set.remove(&del);
                if small_max_set.len() <= big_min_set.len() {
                    small_max_set.insert(add);
                } else {
                    big_min_set.insert(add);
                }
                let small_max = small_max_set.last().copied().unwrap();
                if let Some(big_min) = big_min_set.first().copied() {
                    if small_max > big_min {
                        small_max_set.pop_last();
                        big_min_set.pop_first();
                        small_max_set.insert(big_min);
                        big_min_set.insert(small_max);
                    }
                }
                println!("-> {:?}", small_max_set);
                println!("{:?}", big_min_set);
                l += 1;
                r += 1;
                res[l] = if k % 2 == 0 {
                    (small_max_set.last().copied().unwrap().0 as f64
                        + (big_min_set.first().copied().unwrap().0) as f64)
                        / 2.0
                } else {
                    small_max_set.last().copied().unwrap().0 as f64
                };
            }
            res
        }

        fn mv_average() {
            use std::collections::BTreeSet;
            struct MKAverage {
                small_set: BTreeSet<(i32, usize)>,
                mid_set: BTreeSet<(i32, usize)>,
                big_set: BTreeSet<(i32, usize)>,
                m: usize,
                k: usize,
                nums: Vec<i32>,
                current_sum: i32,
            }

            /**
             * `&self` means the method takes an immutable reference.
             * If you need a mutable reference, change it to `&mut self` instead.
             */
            impl MKAverage {
                fn new(m: i32, k: i32) -> Self {
                    Self {
                        small_set: BTreeSet::new(),
                        mid_set: BTreeSet::new(),
                        big_set: BTreeSet::new(),
                        k: k as usize,
                        m: m as usize,
                        nums: vec![],
                        current_sum: 0,
                    }
                }

                fn add_element(&mut self, num: i32) {
                    if self.nums.len() >= self.m {
                        let del = (
                            self.nums[self.nums.len() - self.m],
                            self.nums.len() - self.m,
                        );
                        self.small_set.remove(&del);
                        self.mid_set.remove(&del);
                        self.big_set.remove(&del);
                        if self.small_set.len() < self.k {
                            self.small_set.insert((num, self.nums.len()));
                        } else if self.big_set.len() < self.k {
                            self.big_set.insert((num, self.nums.len()));
                        } else {
                            self.mid_set.insert((num, self.nums.len()));
                            self.current_sum += num;
                            self.current_sum -= del.0;
                        }
                    } else if self.small_set.len() < self.k {
                        self.small_set.insert((num, self.nums.len()));
                    } else if self.mid_set.len() < self.m - self.k * 2 {
                        self.mid_set.insert((num, self.nums.len()));
                        self.current_sum += num;
                    } else {
                        self.big_set.insert((num, self.nums.len()));
                    }
                    let max_small = self.small_set.last().copied().unwrap();
                    if let Some(min_mid) = self.mid_set.first().copied() {
                        if max_small > min_mid {
                            self.small_set.pop_last();
                            self.mid_set.pop_first();
                            self.small_set.insert(min_mid);
                            self.mid_set.insert(max_small);
                            self.current_sum += max_small.0;
                            self.current_sum -= min_mid.0;
                        }
                        let max_mid = self.mid_set.last().copied().unwrap();
                        if let Some(min_big) = self.big_set.first().copied() {
                            if max_mid > min_big {
                                self.big_set.pop_first();
                                self.mid_set.pop_last();
                                self.mid_set.insert(min_big);
                                self.big_set.insert(max_mid);
                                self.current_sum += min_big.0;
                                self.current_sum -= max_mid.0;
                            }
                            let max_small = self.small_set.last().copied().unwrap();
                            let min_mid = self.mid_set.first().copied().unwrap();
                            if max_small > min_mid {
                                self.small_set.pop_last();
                                self.mid_set.pop_first();
                                self.small_set.insert(min_mid);
                                self.mid_set.insert(max_small);
                                self.current_sum += max_small.0;
                                self.current_sum -= min_mid.0;
                            }
                        }
                    }
                    self.nums.push(num);
                }

                fn calculate_mk_average(&self) -> i32 {
                    if self.nums.len() >= self.m {
                        self.current_sum / self.mid_set.len() as i32
                    } else {
                        -1
                    }
                }
            }
        }
        fn e() {}
    }
}
