use std::{
    cell::RefCell,
    cmp::{Ordering, max},
    collections::{HashMap, HashSet, VecDeque},
    i32,
    ops::Index,
    rc::Rc,
    vec,
};

use log::info;
use rand::Rng;
#[derive(Debug, PartialEq, Eq)]
pub enum NestedInteger {
    Int(i32),
    List(Vec<NestedInteger>),
}
pub fn plus_one(digits: Vec<i32>) -> Vec<i32> {
    let mut digits = digits;
    let mut x;

    let last = digits.last_mut().unwrap();
    *last += 1;
    if *last >= 10 {
        *last -= 10;
        x = 1;
    } else {
        return digits;
    }
    for i in (0..digits.len() - 1).rev() {
        if x == 1 {
            let c = digits.get_mut(i).unwrap();
            *c += 1;
            if *c >= 10 {
                x = 1;
                *c -= 10;
            } else {
                x = 0;
            }
        } else {
            x = 0;
            break;
        }
    }
    if x == 1 {
        digits.splice(0..0, [1]);
    }
    digits
}

pub fn add_binary(mut a: String, mut b: String) -> String {
    unsafe {
        let a = a.as_bytes_mut();
        let b = b.as_bytes_mut();
        a.reverse();
        b.reverse();
        let mut res = vec![];
        let len = usize::max(a.len(), b.len());
        let mut x = 0;
        for i in 0..len {
            let j = if i < a.len() {
                *a.get(i).unwrap() - 48
            } else {
                0
            };
            let k = if i < b.len() {
                *b.get(i).unwrap() - 48
            } else {
                0
            };
            let mut r = j + k + x;
            x = 0;
            if r >= 2 {
                x = 1;
                r -= 2;
            }
            res.push(r + 48);
        }
        if x == 1 {
            res.push(49);
        }
        res.reverse();
        String::from_utf8(res).unwrap()
    }
}

pub fn my_sqrt(x: i32) -> i32 {
    let x = x as u32;
    for i in 1..u32::MAX {
        if i * i > x {
            return (i - 1) as i32;
        }
    }
    return 0;
}

pub fn simplify_path(path: String) -> String {
    let mut components: Vec<String> = vec![];
    let mut p = String::new();

    for ele in path.as_bytes() {
        if *ele == '/' as u8 {
            if !p.is_empty() {
                if p == "." {
                } else if p == ".." {
                    if components.len() != 0 {
                        components.pop().unwrap();
                    }
                } else {
                    components.push(p.clone());
                }

                p.clear();
            }
        } else {
            p.push(*ele as char);
        }
    }
    if !p.is_empty() {
        if p == "." {
        } else if p == ".." {
            if components.len() != 0 {
                components.pop().unwrap();
            }
        } else {
            components.push(p.clone());
        }
    }
    let mut res = components.join("/");
    res.insert(0, '/');
    res
}

pub fn min_distance(s: String, t: String) -> i32 {
    let (n, m) = (s.len(), t.len());
    let mut dp = vec![vec![0; m + 1]; n + 1];
    // 状态转移：首行首列
    for i in 1..=n {
        dp[i][0] = i as i32;
    }
    for j in 1..=m {
        dp[0][j] = j as i32;
    }
    println!("{:#?}", dp);
    if n == 0 || m == 0 {
        return dbg!(dp[n][m]);
    }
    // 状态转移：其余行和列
    for i in 1..=n {
        for j in 1..=m {
            if s.chars().nth(i - 1) == t.chars().nth(j - 1) {
                // 若两字符相等，则直接跳过此两字符
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                // 最少编辑步数 = 插入、删除、替换这三种操作的最少编辑步数 + 1
                dp[i][j] =
                    std::cmp::min(std::cmp::min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
            }
        }
    }
    dp[n][m]
}
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}
pub struct A {}
#[allow(unused)]
impl A {
    pub fn dfs_is_symmetric(
        left: Option<&Rc<RefCell<TreeNode>>>,
        right: Option<&Rc<RefCell<TreeNode>>>,
    ) -> bool {
        if let (Some(l), Some(r)) = (left, right) {
            if l.borrow().val != r.borrow().val {
                return false;
            } else {
                let a = Self::dfs_is_symmetric(l.borrow().left.as_ref(), r.borrow().right.as_ref());
                if !a {
                    return false;
                }
                let b = Self::dfs_is_symmetric(l.borrow().right.as_ref(), r.borrow().left.as_ref());
                if !b {
                    return false;
                }
                return true;
            }
        } else if let (None, None) = (left, right) {
            return true;
        } else {
            return false;
        }
    }
    pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        if let Some(root) = root {
            Self::dfs_is_symmetric(root.borrow().left.as_ref(), root.borrow().right.as_ref())
        } else {
            true
        }
    }
    pub fn dfs_level_order(
        root: Option<&Rc<RefCell<TreeNode>>>,
        deep: usize,
        res: &mut Vec<Vec<i32>>,
    ) {
        if let Some(root) = root {
            if res.len() <= deep {
                res.push(vec![]);
            }
            res[deep].push(root.borrow().val);
            Self::dfs_level_order(root.borrow().left.as_ref(), deep + 1, res);
            Self::dfs_level_order(root.borrow().right.as_ref(), deep + 1, res);
        }
    }
    pub fn level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut res = vec![];
        Self::dfs_level_order(root.as_ref(), 0, &mut res);
        res
    }

    pub fn level_order2(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut queue = VecDeque::new();
        let mut res = Vec::new();
        if let Some(root) = root {
            queue.push_back(Rc::clone(&root));
            while !queue.is_empty() {
                let curremt_queue_size = queue.len();
                let mut current_nodes = vec![];
                for _ in 0..curremt_queue_size {
                    let tree = queue.pop_front().unwrap();
                    current_nodes.push(tree.borrow().val);
                    let borrow = tree.borrow();
                    if let Some(left) = borrow.left.as_ref() {
                        queue.push_back(Rc::clone(left));
                    }
                    if let Some(right) = borrow.right.as_ref() {
                        queue.push_back(Rc::clone(right));
                    }
                }
                res.push(current_nodes);
            }
        }
        return res;
    }
    pub fn zigzag_level_order(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut queue = VecDeque::new();
        let mut res = Vec::new();
        if let Some(root) = root {
            let mut from_left_to_right = true;
            queue.push_back(Rc::clone(&root));
            while !queue.is_empty() {
                let curremt_queue_size = queue.len();
                let mut current_nodes = vec![];
                for _ in 0..curremt_queue_size {
                    let tree = queue.pop_front().unwrap();
                    current_nodes.push(tree.borrow().val);
                    let borrow = tree.borrow();
                    if let Some(left) = borrow.left.as_ref() {
                        queue.push_back(Rc::clone(left));
                    }
                    if let Some(right) = borrow.right.as_ref() {
                        queue.push_back(Rc::clone(right));
                    }
                }
                if from_left_to_right {
                    res.push(current_nodes);
                } else {
                    let s = current_nodes.into_iter().rev().collect::<Vec<i32>>();
                    res.push(s);
                }
                from_left_to_right = !from_left_to_right
            }
        }
        return res;
    }

    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(r: Option<&Rc<RefCell<TreeNode>>>, deep: i32, res: &mut i32) {
            if let Some(r) = r {
                if deep + 1 > *res {
                    *res = deep + 1
                }
                dfs(r.borrow().left.as_ref(), deep + 1, res);
                dfs(r.borrow().right.as_ref(), deep + 1, res);
            }
        }
        let mut res = 0;
        dfs(root.as_ref(), 0, &mut res);
        res
    }

    pub fn build_tree_with_ref(preorder: &[i32], inorder: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        if preorder.len() == 0 {
            return None;
        }

        let root = preorder.first().unwrap();

        let mut seperator = 0;
        for (offset, ele) in inorder.iter().enumerate() {
            if *root == *ele {
                seperator = offset;
                break;
            }
        }
        let pre_left = &preorder[1..seperator + 1];
        // for i in 1..seperator + 1 {
        //     pre_left.push(preorder[i]);
        // }

        let pre_right = &preorder[seperator + 1..];
        // for i in seperator + 1..preorder.len() {
        //     pre_right.push(preorder[i]);
        // }
        let ino_left = &inorder[0..seperator];
        // for i in 0..seperator {
        //     ino_left.push(inorder[i]);
        // }
        let ino_right = &inorder[seperator + 1..];
        // for i in seperator + 1..inorder.len() {
        //     ino_right.push(inorder[i]);
        // }

        let left = Self::build_tree_with_ref(pre_left, ino_left);
        let right = Self::build_tree_with_ref(pre_right, ino_right);
        let mut root_node = Some(Rc::new(RefCell::new(TreeNode::new(*root))));
        let r = root_node.as_mut().unwrap();
        r.borrow_mut().left = left;
        r.borrow_mut().right = right;
        root_node
    }

    pub fn build_tree(preorder: Vec<i32>, inorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        Self::build_tree_with_ref(&preorder, &inorder)
    }

    fn build_tree_with_ref2(inorder: &[i32], postorder: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        if inorder.len() == 0 {
            return None;
        }

        println!("{:?}", inorder);
        println!("{:?}", postorder);
        let root = postorder.last().unwrap();

        let mut seperator = 0;
        for (offset, ele) in inorder.iter().enumerate() {
            if *root == *ele {
                seperator = offset;
                break;
            }
        }
        let post_left = &postorder[0..seperator];

        let post_right = &postorder[seperator + 1..postorder.len() - 1];
        let ino_left = &inorder[0..seperator];
        let ino_right = &inorder[seperator + 1..];

        let left = Self::build_tree_with_ref2(ino_left, post_left);
        let right = Self::build_tree_with_ref2(ino_right, post_right);
        let mut root_node = Some(Rc::new(RefCell::new(TreeNode::new(*root))));
        let r = root_node.as_mut().unwrap();
        r.borrow_mut().left = left;
        r.borrow_mut().right = right;
        root_node
    }
    pub fn build_tree2(inorder: Vec<i32>, postorder: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        Self::build_tree_with_ref2(&inorder, &postorder)
    }

    pub fn level_order3(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Vec<i32>> {
        let mut queue = VecDeque::new();
        let mut res = Vec::new();
        if let Some(root) = root {
            queue.push_back(Rc::clone(&root));
            while !queue.is_empty() {
                let curremt_queue_size = queue.len();
                let mut current_nodes = vec![];
                for _ in 0..curremt_queue_size {
                    let tree = queue.pop_front().unwrap();
                    current_nodes.push(tree.borrow().val);
                    let borrow = tree.borrow();
                    if let Some(left) = borrow.left.as_ref() {
                        queue.push_back(Rc::clone(left));
                    }
                    if let Some(right) = borrow.right.as_ref() {
                        queue.push_back(Rc::clone(right));
                    }
                }
                res.push(current_nodes);
            }
        }
        return res.into_iter().rev().collect();
        // return res;
    }
    pub fn sorted_array_to_bst_with_ref(nums: &[i32]) -> Option<Rc<RefCell<TreeNode>>> {
        if nums.len() == 0 {
            return None;
        }
        let mid_index = nums.len() / 2;
        let left = &nums[0..mid_index];
        let right = &nums[mid_index + 1..];
        let mut root_node = Some(Rc::new(RefCell::new(TreeNode::new(nums[mid_index]))));
        let left_node = Self::sorted_array_to_bst_with_ref(left);
        let right_node = Self::sorted_array_to_bst_with_ref(right);
        let r = root_node.as_mut().unwrap();
        r.borrow_mut().left = left_node;
        r.borrow_mut().right = right_node;
        root_node
    }

    pub fn sorted_array_to_bst(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        Self::sorted_array_to_bst_with_ref(&nums)
    }

    pub fn sorted_list_to_bst(head: Option<Box<ListNode>>) -> Option<Rc<RefCell<TreeNode>>> {
        let mut nums = vec![];
        let mut head = head.as_ref();
        while let Some(p) = head {
            nums.push(p.val);
            head = p.next.as_ref()
        }
        return Self::sorted_array_to_bst(nums);
    }

    //110.
    pub fn is_balanced(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn dfs_is_balanced(root: Option<&Rc<RefCell<TreeNode>>>, res: &mut bool) -> u32 {
            if !*res {
                return 0;
            }
            if let Some(root) = root {
                let left = dfs_is_balanced(root.borrow().left.as_ref(), res);
                let right = dfs_is_balanced(root.borrow().right.as_ref(), res);
                if u32::abs_diff(right, left) >= 2 {
                    *res = false;
                }
                return 1 + u32::max(left, right);
            } else {
                return 0;
            }
        }
        let mut res = true;
        // if let Some(root) = root {
        dfs_is_balanced(root.as_ref(), &mut res);
        // }
        res
    }

    //111.
    pub fn min_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs_min_depth(root: Option<&Rc<RefCell<TreeNode>>>, deep: i32, res: &mut i32) {
            if let Some(root) = root {
                if root.borrow().left == None && root.borrow().right == None {
                    if *res > deep {
                        *res = deep
                    } else {
                        dfs_min_depth(root.borrow().left.as_ref(), deep + 1, res);
                        dfs_min_depth(root.borrow().right.as_ref(), deep + 1, res);
                    }
                }
            }
        }
        let mut res: i32 = i32::MAX;
        dfs_min_depth(root.as_ref(), 0, &mut res);
        return res;
    }
    //112.
    pub fn has_path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> bool {
        fn dfs_has_path_sum(
            root: Option<&Rc<RefCell<TreeNode>>>,
            current: i32,
            target_sum: i32,
            res: &mut bool,
        ) {
            if *res {
                return;
            }
            if let Some(root) = root {
                if root.borrow().left.is_none()
                    && root.borrow().right.is_none()
                    && current + root.borrow().val == target_sum
                {
                    *res = true
                } else {
                    dfs_has_path_sum(
                        root.borrow().left.as_ref(),
                        current + root.borrow().val,
                        target_sum,
                        res,
                    );
                    dfs_has_path_sum(
                        root.borrow().right.as_ref(),
                        current + root.borrow().val,
                        target_sum,
                        res,
                    );
                }
            }
        }
        let mut res = false;
        if root.is_some() {
            dfs_has_path_sum(root.as_ref(), 0, target_sum, &mut res);
        }
        res
    }
    // 113.
    pub fn path_sum(root: Option<Rc<RefCell<TreeNode>>>, target_sum: i32) -> Vec<Vec<i32>> {
        fn dfs_path_sum(
            root: Option<&Rc<RefCell<TreeNode>>>,
            current: &mut Vec<i32>,
            res: &mut Vec<Vec<i32>>,
            target_sum: i32,
        ) {
            if let Some(root) = root {
                let root = root.borrow();
                current.push(root.val);
                let sum = current.iter().sum::<i32>();
                if sum < target_sum {
                    current.pop();
                    return;
                }
                if root.left.is_none() && root.right.is_none() && sum == target_sum {
                    res.push(current.clone());
                } else {
                    dfs_path_sum(root.left.as_ref(), current, res, target_sum);
                    dfs_path_sum(root.right.as_ref(), current, res, target_sum);
                }
                current.pop();
            }
        }
        let mut res = vec![];
        let mut current = vec![];
        dfs_path_sum(root.as_ref(), &mut current, &mut res, target_sum);
        res
    }
    //114. using recursion to play with Linked things
    pub fn flatten(root: &mut Option<Rc<RefCell<TreeNode>>>) {
        fn set_last(root: Option<&Rc<RefCell<TreeNode>>>, last: Option<Rc<RefCell<TreeNode>>>) {
            if let Some(root) = root {
                if root.borrow().right.is_none() {
                    root.borrow_mut().right = last
                } else {
                    set_last(root.borrow().right.as_ref(), last);
                }
            }
        }
        fn dfs_flatten(root: Option<&Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
            if let Some(root) = root {
                let mut flatend_left = dfs_flatten(root.borrow().left.as_ref());
                let flatend_right = dfs_flatten(root.borrow().right.as_ref());
                let p = Rc::clone(root);

                if let Some(r) = flatend_right.as_ref() {
                    r.borrow_mut().left = None
                }

                if let Some(l) = flatend_left.as_ref() {
                    set_last(Some(l), flatend_right);
                    l.borrow_mut().left = None;
                } else {
                    flatend_left = flatend_right
                }

                p.borrow_mut().right = flatend_left;
                p.borrow_mut().left = None;
                Some(p)
            } else {
                None
            }
        }
        dfs_flatten(root.as_ref());
    }

    pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
        let num_rows = num_rows as usize;
        let mut res = vec![];
        for i in 0..num_rows {
            let p = vec![1; i + 1];
            res.push(p);
            if res[i].len() > 2 {
                for j in 1..res[i].len() - 1 {
                    res[i][j] = res[i - 1][j - 1] + res[i - 1][j];
                }
            }
        }
        res
    }
    pub fn get_row(row_index: i32) -> Vec<i32> {
        let e = Self::generate(row_index).pop().unwrap();
        e
    }
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut dp = vec![vec![0; prices.len()]; 2];
        dp[0][0] = -prices[0];
        dp[1][0] = prices[0];

        for i in 1..prices.len() {
            let profit = prices[i] - dp[1][i - 1];
            if profit < 0 {
                dp[0][i] = -prices[i];
                dp[1][i] = prices[i];
            } else {
                if profit < dp[0][i - 1] {
                    dp[0][i] = dp[0][i - 1]
                } else {
                    dp[0][i] = profit
                }
                dp[1][i] = dp[1][i - 1];
            }
        }
        let res = dp[0].iter().max().unwrap();
        if *res > 0 { *res } else { 0 }
    }

    pub fn max_profit2(prices: Vec<i32>) -> i32 {
        // 0 : own 1:not own
        let mut dp = vec![vec![0; prices.len()]; 2];
        dp[0][0] = -prices.first().unwrap();
        dp[1][0] = 0;
        for i in 1..prices.len() {
            // own = not own - currentPrice, own
            dp[0][i] = i32::max(dp[0][i - 1], dp[1][i - 1] - prices[i]);
            // not = own + currentPrice , not
            dp[1][i] = i32::max(dp[1][i - 1], dp[0][i - 1] + prices[i])
        }
        return i32::max(*dp[0].last().unwrap(), *dp[1].last().unwrap());
    }

    //123
    pub fn max_profit3(prices: Vec<i32>) -> i32 {
        let mut buy1 = -prices[0];
        let mut buy2 = -prices[0];
        let mut sale1 = 0;
        let mut sale2 = 0;
        for i in 1..prices.len() {
            buy1 = i32::max(buy1, -prices[i]);
            sale1 = i32::max(sale1, buy1 + prices[i]);
            buy2 = i32::max(buy2, sale1 - prices[i]);
            sale2 = i32::max(sale2, buy2 + prices[i]);
        }
        sale2
    }
    //125
    pub fn is_palindrome(s: String) -> bool {
        let res = s
            .to_lowercase()
            .chars()
            .filter(|x| x.is_ascii_alphanumeric())
            .collect::<Vec<char>>();
        for i in 0..res.len() / 2 {
            if res[i] != res[res.len() - i - 1] {
                return false;
            }
        }
        true
    }
    //128
    pub fn longest_consecutive(mut nums: Vec<i32>) -> i32 {
        if nums.is_empty() {
            return 0;
        }
        nums.sort();
        let mut res = 1;
        let mut x = 1;
        for i in 0..nums.len() - 1 {
            if nums[i] == nums[i + 1] {
                continue;
            }
            if nums[i] + 1 == nums[i + 1] {
                x += 1;
            } else {
                if res < x {
                    res = x;
                }
                x = 1;
            }
        }
        if res > x { res } else { x }
    }

    pub fn sum_numbers(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs_sum_numbers(
            root: &Rc<RefCell<TreeNode>>,
            current: &mut Vec<i32>,
            res: &mut Vec<i32>,
        ) {
            current.push(root.borrow().val);
            if root.borrow().left.is_none() && root.borrow().right.is_none() {
                let mut sum = 0;
                let mut step = 1;
                for i in current.iter().rev() {
                    sum += *i * step;
                    step *= 10;
                }
                res.push(sum);
            }
            if root.borrow().left.is_some() {
                dfs_sum_numbers(root.borrow().left.as_ref().unwrap(), current, res);
            }
            if root.borrow().right.is_some() {
                dfs_sum_numbers(root.borrow().right.as_ref().unwrap(), current, res);
            }
            current.pop().unwrap();
        }
        let mut current = vec![];
        let mut res = vec![];
        dfs_sum_numbers(root.as_ref().unwrap(), &mut current, &mut res);
        return res.iter().sum();
    }

    pub fn solve(board: &mut Vec<Vec<char>>) {
        fn dfs_solve(board: &mut Vec<Vec<char>>, i: usize, j: usize) {
            if i < board.len() && j < board.first().unwrap().len() {
                if board[i][j] == 'O' {
                    board[i][j] = 'N';
                    dfs_solve(board, i - 1, j);
                    dfs_solve(board, i + 1, j);
                    dfs_solve(board, i, j + 1);
                    dfs_solve(board, i, j - 1);
                }
            }
        }
        for i in 0..board.len() {
            dfs_solve(board, i, 0);
            dfs_solve(board, i, board.first().unwrap().len() - 1);
        }
        for j in 0..board.first().unwrap().len() {
            dfs_solve(board, 0, j);
            dfs_solve(board, board.len() - 1, j);
        }
        for i in 0..board.len() {
            for j in 0..board.first().unwrap().len() {
                if board[i][j] == 'O' {
                    board[i][j] = 'X';
                } else if board[i][j] == 'N' {
                    board[i][j] = 'O';
                }
            }
        }
        //131
        pub fn partition(s: String) -> Vec<Vec<String>> {
            fn is_huiwen(s: &str) -> bool {
                let e = s.as_bytes();
                for i in 0..e.len() / 2 {
                    if e[i] != e[e.len() - 1 - i] {
                        return false;
                    }
                }
                return true;
            }
            fn dfs_parition(
                s: &str,
                start: usize,
                current: &mut Vec<String>,
                res: &mut Vec<Vec<String>>,
            ) {
                if start >= s.len() {
                    res.push(current.clone());
                    return;
                }
                for i in start + 1..=s.len() {
                    let ss = &s[start..i];
                    if is_huiwen(ss) {
                        current.push(ss.to_string());
                        println!("{:?}", current);
                        dfs_parition(s, i, current, res);
                        current.pop().unwrap();
                    }
                }
            }
            let mut current = vec![];
            let mut res = vec![];
            dfs_parition(&s, 0, &mut current, &mut res);
            return res;
        }
    }

    pub fn min_cut(s: String) -> i32 {
        let s = s.as_bytes();
        let mut dp = vec![vec![true; s.len()]; s.len()];
        for step in 1..s.len() {
            for i in 0.. {
                if i + step >= s.len() {
                    break;
                }
                let n = i;
                let m = i + step;
                if n + 1 > m - 1 {
                    dp[n][m] = s[n] == s[m];
                } else {
                    dp[n][m] = dp[n + 1][m - 1] && s[n] == s[m];
                }
            }
        }
        let mut dp2 = vec![i32::MAX; s.len()];
        for i in 0..s.len() {
            if dp[0][i] {
                dp2[i] = 0;
            } else {
                for j in 0..i {
                    if dp[j + 1][i] {
                        dp2[i] = i32::min(dp2[i], dp2[j] + 1);
                    }
                }
            }
        }
        *dp2.last().unwrap()
    }
    //134
    pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        if gas.len() == 1 {
            if gas.first().unwrap() >= cost.first().unwrap() {
                return 0;
            }
        }
        let mut current_gas;
        let mut current_station;
        for start_idx in 0..gas.len() {
            current_gas = gas[start_idx];
            let start_station = start_idx;
            current_station = start_idx;
            let mut ok = true;
            if gas[start_idx] == cost[start_idx] {
                continue;
            }
            // drive
            for _ in 0..gas.len() {
                let current_cost = cost[current_station];
                current_station += 1;
                if current_station == gas.len() {
                    current_station %= gas.len();
                }
                if current_gas - current_cost < 0 {
                    ok = false;
                    break;
                } else {
                    current_gas = current_gas - current_cost + gas[current_station];
                }
            }
            if ok {
                return start_station as i32;
            }
        }
        return -1;
    }

    //136
    pub fn single_number(nums: Vec<i32>) -> i32 {
        let mut map = HashMap::new();
        for ele in nums {
            if map.contains_key(&ele) {
                let e = map.get_mut(&ele).unwrap();
                *e += 1;
            } else {
                map.insert(ele, 1);
            }
        }
        for (k, v) in map {
            if v == 1 {
                return k;
            }
        }
        0
    }
    //139
    pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
        let mut dp = vec![false; s.len()];
        let word_dict_ref: Vec<&str> = word_dict.iter().map(|x| &x[..]).collect();
        for i in 0..s.len() {
            if word_dict_ref.contains(&&s[0..=i]) {
                dp[i] = true;
            } else {
                for j in 0..i {
                    if dp[j] == true && word_dict_ref.contains(&&s[j + 1..=i]) {
                        dp[i] = true
                    }
                }
            }
        }
        *dp.last().unwrap()
    }

    pub fn word_break2(s: String, word_dict: Vec<String>) -> Vec<String> {
        let mut res = vec![];
        let mut dp = vec![false; s.len()];
        let word_dict_ref: Vec<&str> = word_dict.iter().map(|x| &x[..]).collect();
        for i in 0..s.len() {
            if word_dict_ref.contains(&&s[0..=i]) {
                dp[i] = true;
                if i > 0 {
                    if dp[i - 1] == true {
                        res.pop().unwrap();
                    }
                }
                res.push(s[0..=i].to_string());
            } else {
                for j in 0..i {
                    if dp[j] == true && word_dict_ref.contains(&&s[j + 1..=i]) {
                        dp[i] = true;
                        res.push(s[j + 1..=i].to_string());
                        break;
                    }
                }
            }
        }
        if *dp.last().unwrap() { res } else { vec![] }
    }
    //143
    pub fn reorder_list(head: &mut Option<Box<ListNode>>) {
        let mut nodes = HashMap::new();
        fn to_map(
            head: &mut Option<Box<ListNode>>,
            nodes: &mut HashMap<usize, Option<Box<ListNode>>>,
            n: usize,
        ) {
            if let Some(head) = head {
                let mut next = head.next.take();

                to_map(&mut next, nodes, n + 1);
                if next.is_some() {
                    nodes.insert(n, next);
                }
            }
        }
        to_map(head, &mut nodes, 1);
        for ele in nodes.iter() {
            if let Some(e) = ele.1 {
                println!("{}-{:?}", ele.0, e.val);
            } else {
                println!("{:?} is none", ele.0);
            }
        }
        let len = nodes.len() + 1;
        fn resume(
            head: &mut Option<Box<ListNode>>,
            mut nodes: HashMap<usize, Option<Box<ListNode>>>,
            n: usize,
            len: usize,
            real_len: usize,
        ) {
            if n > len {
                return;
            }
            if let Some(head) = head {
                if n == 0 {
                    let mut next = nodes.remove(&n).unwrap();
                    resume(&mut next, nodes, n + 1, len, real_len);
                    head.next = next;
                } else {
                    let mut next = nodes.remove(&n).unwrap();
                    let mut next_next = nodes.remove(&(real_len - n)).unwrap();
                    resume(&mut next_next, nodes, n + 1, len, real_len);
                    next.as_mut().unwrap().next = next_next;
                    head.next = next;
                }
            }
        }
        resume(head, nodes, 0, len / 2, len - 1);
    }
    pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut res = vec![];
        fn dfs(root: Option<&Rc<RefCell<TreeNode>>>, res: &mut Vec<i32>) {
            if let Some(root) = root {
                res.push(root.borrow().val);
                dfs(root.borrow().left.as_ref(), res);
                dfs(root.borrow().right.as_ref(), res);
            }
        }
        dfs(root.as_ref(), &mut res);
        res
    }
    //148
    fn e() {
        struct CacheValue {
            value: i32,
            life: i32,
        }
        struct LRUCache {
            store: HashMap<i32, CacheValue>,
            capacity: i32,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl LRUCache {
            fn new(capacity: i32) -> Self {
                Self {
                    store: HashMap::with_capacity(capacity as usize),
                    capacity: capacity,
                }
            }

            fn get(&mut self, key: i32) -> i32 {
                for x in self.store.values_mut() {
                    x.life -= 1;
                }
                if let Some(e) = self.store.get_mut(&key) {
                    e.life = 0;
                    //
                    return e.value;
                } else {
                    -1
                }
            }

            fn put(&mut self, key: i32, value: i32) {
                for x in self.store.values_mut() {
                    x.life -= 1;
                }
                if self.store.contains_key(&key) {
                    self.store.insert(key, CacheValue { value, life: 0 });
                    return;
                }
                if self.store.len() >= self.capacity as usize {
                    let mut key_to_rm = 0;
                    let mut min_life = 0;
                    //get the smallest life
                    for (k, v) in self.store.iter() {
                        if v.life < min_life {
                            min_life = v.life;
                            key_to_rm = *k;
                        }
                    }

                    self.store.remove(&key_to_rm).unwrap();
                    self.store.insert(key, CacheValue { value, life: 0 });
                } else {
                    self.store.insert(key, CacheValue { value, life: 0 });
                }
            }
        }
    }
    //149
    pub fn insertion_sort_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        if head.is_none() {
            return None;
        }
        let mut res = Box::new(ListNode::new(9));
        fn dfs(head: Option<Box<ListNode>>, res: &mut Box<ListNode>) {
            if let Some(mut next) = head {
                let nn = next.next.take();
                dfs_insert(res, next);
                dfs(nn, res);
            }
        }
        fn dfs_insert(head: &mut Box<ListNode>, mut value: Box<ListNode>) {
            if head.next.is_none() {
                head.next = Some(value);
                return;
            } else if head.next.as_ref().unwrap().val > value.val {
                let next_next = head.next.take();
                value.next = next_next;
                head.next = Some(value)
            } else {
                let next = head.next.as_mut().unwrap();
                dfs_insert(next, value);
            }
        }
        dfs(head, &mut res);
        res.next
    }
    pub fn eval_rpn(mut tokens: Vec<String>) -> i32 {
        let mut stack = vec![];
        tokens.reverse();
        while let Some(next) = tokens.pop() {
            if next == "+" {
                let right = stack.pop().unwrap();
                let left = stack.pop().unwrap();
                stack.push(right + left);
            } else if next == "-" {
                let right = stack.pop().unwrap();
                let left = stack.pop().unwrap();
                stack.push(left - right);
            } else if next == "*" {
                let right = stack.pop().unwrap();
                let left = stack.pop().unwrap();
                stack.push(left * right);
            } else if next == "/" {
                let right = stack.pop().unwrap();
                let left = stack.pop().unwrap();
                stack.push(left / right);
            } else {
                stack.push(next.parse::<i32>().unwrap());
            }
        }
        stack.pop().unwrap()
    }
    //151
    pub fn reverse_words(mut s: String) -> String {
        let mut buf = VecDeque::new();
        //hello
        //hello
        while let Some(c) = s.pop() {
            buf.push_front(c);
        }

        let mut buf_str = String::new();
        let mut res = vec![];
        while let Some(c) = buf.pop_front() {
            if c == ' ' {
                if !buf_str.is_empty() {
                    res.push(buf_str.clone());
                }
                buf_str.clear();
                continue;
            } else {
                buf_str.push(c);
            }
        }
        if !buf_str.is_empty() {
            res.push(buf_str);
        }
        res.reverse();
        res.join(" ")
    }
    //152
    pub fn max_product(nums: Vec<i32>) -> i32 {
        let mut dp = vec![0; nums.len()];
        let mut res = i32::MIN;
        for i in 0..nums.len() {
            dp.push(nums[i]);
            if res < nums[i] {
                res = nums[i]
            }
            for j in i + 1..nums.len() {
                dp[j] = dp[j - 1] * nums[j];
                if dp[j] > res {
                    res = dp[j];
                }
            }
            dp.clear();
            dp.reserve(nums.len());
        }
        res
    }
    //153
    pub fn find_min(nums: Vec<i32>) -> i32 {
        // the nums is a sorted array ,and it is rotated
        // 1,2,3,4,5,6
        // 4,5,6,1,2,3
        // find the min one
        let mut res = i32::MAX;
        let mut left = 0;
        let mut right = nums.len() - 1;
        while left <= right {
            let mid = (left + right) / 2;
            if nums[mid] < nums[left] {
                right = mid + 1;
            } else if nums[mid] > nums[right] {
                left = mid + 1;
            } else if left == 0 {
                res = nums[left];
            } else if nums[left] > nums[left - 1] {
                res = nums[left]
            }
        }
        res
    }
    //155
    fn min_stack() {
        struct MinValue {
            value: i32,
            count: i32,
        }
        struct MinStack {
            stack: Vec<i32>,
            min_value: Option<MinValue>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl MinStack {
            fn new() -> Self {
                Self {
                    stack: vec![],
                    min_value: None,
                }
            }

            fn push(&mut self, val: i32) {
                if self.min_value.is_none() {
                    self.min_value = Some(MinValue {
                        value: val,
                        count: 1,
                    })
                } else if val == self.min_value.as_ref().unwrap().value {
                    self.min_value.as_mut().unwrap().count += 1;
                } else if val < self.min_value.as_ref().unwrap().value {
                    self.min_value.as_mut().unwrap().count = 1;
                    self.min_value.as_mut().unwrap().value = val;
                }
                self.stack.push(val);
            }

            fn pop(&mut self) {
                let last = self.stack.pop().unwrap();
                if last == self.get_min() {
                    self.min_value.as_mut().unwrap().count -= 1;
                    if self.min_value.as_ref().unwrap().count == 0 {
                        // search new
                        let mut new_min = MinValue {
                            value: i32::MAX,
                            count: 0,
                        };
                        for ele in self.stack.iter() {
                            if *ele < new_min.value {
                                new_min.count = 0;
                                new_min.value = *ele;
                            } else if *ele == new_min.value {
                                new_min.count += 1;
                            }
                        }
                        self.min_value = Some(new_min)
                    }
                }
            }

            fn top(&self) -> i32 {
                *self.stack.last().unwrap()
            }

            fn get_min(&self) -> i32 {
                self.min_value.as_ref().unwrap().value
            }
        }
    }
    //164
    pub fn find_peak_element(nums: Vec<i32>) -> i32 {
        for (idx, value) in nums.iter().enumerate() {
            let left = if idx == 0 { i32::MIN } else { nums[idx - 1] };
            let right = if idx == nums.len() - 1 {
                i32::MIN
            } else {
                nums[idx + 1]
            };
            if *value > left && *value > right {
                return idx as i32;
            }
        }
        0
    }
    pub fn maximum_gap(mut nums: Vec<i32>) -> i32 {
        nums.sort();
        let mut max = 0;

        for i in 1..nums.len() {
            let interval = nums[i] - nums[i - 1];
            if interval > max {
                max = interval;
            }
        }
        max
    }
    //165
    pub fn compare_version(version1: String, version2: String) -> i32 {
        let version1 = version1
            .split(".")
            .map(|x| x.parse::<i32>().unwrap())
            .collect::<Vec<i32>>();
        let version2 = version2
            .split(".")
            .map(|x| x.parse::<i32>().unwrap())
            .collect::<Vec<i32>>();
        let mut idx = 0;
        loop {
            let v1 = version1.get(idx);
            let v2 = version2.get(idx);

            if v1.is_some() && v2.is_some() {
                let v1 = v1.unwrap();
                let v2 = v2.unwrap();
                if v1 == v2 {
                } else if v1 > v2 {
                    return 1;
                } else {
                    return -1;
                }
            }

            if v1.is_none() && v2.is_none() {
                return 0;
            }
            if v1.is_none() && v2.is_some() {
                if let Some(v2) = v2 {
                    if *v2 != 0 {
                        return -1;
                    }
                }
            }

            if v2.is_none() && v1.is_some() {
                if let Some(v1) = v1 {
                    if *v1 != 0 {
                        return -1;
                    }
                }
            }
            idx += 1
        }
    }
    //166
    pub fn fraction_to_decimal(numerator: i32, denominator: i32) -> String {
        if numerator == 0 {
            return "0".to_string();
        }
        let mut is_negtive = false;
        if numerator < 0 || denominator < 0 {
            is_negtive = true;
        }
        if numerator < 0 && denominator < 0 {
            is_negtive = false;
        }
        let numerator = (numerator as i64).abs();
        let denominator = (denominator as i64).abs();
        let x = numerator / denominator;
        let mut vec = vec![];
        let mut left = numerator;
        let mut s = String::new();

        loop {
            left = left % denominator;
            if left == 0 {
                break;
            }
            if vec.contains(&left) {
                let mut x = 0;
                for (idx, e) in vec.iter().enumerate() {
                    if *e == left {
                        x = idx;
                        break;
                    }
                }
                s.insert(x, '(');
                s.push(')');
                break;
            }

            vec.push(left);
            left *= 10;
            let y = left / denominator;
            s.push_str(&y.to_string());
        }
        if s.is_empty() {
            let res = format!("{}{}", if is_negtive { "-" } else { "" }, x);
            res
        } else {
            let res = format!("{}{}.{}", if is_negtive { "-" } else { "" }, x, s);
            res
        }
    }
    //167
    pub fn two_sum(numbers: Vec<i32>, target: i32) -> Vec<i32> {
        for i in 0..numbers.len() - 1 {
            for j in i + 1..numbers.len() {
                if numbers[i] + numbers[j] == target {
                    return vec![(i + 1) as i32, (j + 1) as i32];
                } else if numbers[i] + numbers[j] > target {
                    break;
                }
            }
        }
        vec![]
    }
    //168
    pub fn convert_to_title(mut column_number: i32) -> String {
        if column_number == 26 {
            return "Z".to_string();
        }
        let mut res = vec![];
        while column_number != 0 {
            let c = column_number % 26;
            println!("{:?}", (c as u8 + 64) as char);
            if c == 0 {
                res.push(b'Z');
            } else {
                res.push((c as u8 + 64) as u8);
            }

            column_number /= 26;
            if column_number == 26 {
                println!("{:?}", "Z");
                res.push(b'Z');
                break;
            }
        }
        res.reverse();
        String::from_utf8(res).unwrap()
    }
    //171
    pub fn title_to_number(mut column_title: String) -> i32 {
        let mut step: i32 = 1;
        let mut res: i32 = 0;
        while let Some(next) = column_title.pop() {
            let next_u8 = (next as u8 - 64) as i32;
            res += next_u8 * step;
            step *= 26;
        }
        res
    }
    //172
    pub fn trailing_zeroes(n: i32) -> i32 {
        let mut step = 5;
        let mut res = 0;
        while step <= n {
            res += n / step;
            step *= 5;
        }
        res
    }
    //173
    fn eleee() {
        struct BSTIterator {
            nodes: Vec<i32>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl BSTIterator {
            fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
                fn dfs(root: Option<Rc<RefCell<TreeNode>>>, nodes: &mut Vec<i32>) {
                    if let Some(root) = root {
                        let left = root.borrow_mut().left.take();
                        dfs(left, nodes);
                        nodes.push(root.borrow().val);
                        let right = root.borrow_mut().right.take();
                        dfs(right, nodes);
                    }
                }
                let mut nodes = vec![];
                dfs(root, &mut nodes);
                nodes.reverse();
                Self { nodes }
            }

            fn next(&mut self) -> i32 {
                self.nodes.pop().unwrap()
            }

            fn has_next(&self) -> bool {
                !self.nodes.is_empty()
            }
        }
    }
    //179
    pub fn largest_number(mut nums: Vec<i32>) -> String {
        nums.sort_by(|a, b| {
            let mut e = String::new();
            e.push_str(&a.to_string());
            e.push_str(&b.to_string());
            let mut x = String::new();
            x.push_str(&b.to_string());
            x.push_str(&a.to_string());
            let left = e.parse::<i32>().unwrap();
            let right = x.parse::<i32>().unwrap();
            if left > right {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        println!("{:?}", nums);
        if *nums.first().unwrap() == 0 {
            return "0".to_string();
        }
        let mut res = String::new();
        for e in nums {
            res.push_str(&e.to_string());
        }
        res
    }
    //187
    pub fn find_repeated_dna_sequences(s: String) -> Vec<String> {
        if s.len() < 10 {
            return vec![];
        }
        let mut map = HashMap::new();

        for i in 0..s.len() - 9 {
            let sub = &s[i..i + 10];
            if map.contains_key(&sub) {
                *map.get_mut(&sub).unwrap() += 1;
            } else {
                map.insert(sub, 0);
            }
        }
        let mut res = vec![];
        for (k, v) in map {
            if v > 1 {
                res.push(k.to_string());
            }
        }
        res
    }
    //189
    pub fn rotate(nums: &mut Vec<i32>, k: i32) {
        if k > nums.len() as i32 {
            for _ in 0..k {
                let last = nums.pop().unwrap();
                nums.insert(0, last);
            }
        } else {
            let mut p = vec![];
            for _ in 0..k {
                if let Some(next) = nums.pop() {
                    p.push(next);
                } else {
                    break;
                }
            }
            p.reverse();
            p.append(nums);
            *nums = p;
        }
    }
    //191
    pub fn hamming_weight(mut n: i32) -> i32 {
        let mut res = 0;
        while n != 0 {
            let x = n % 2;
            if x == 1 {
                res += 1;
            }
            n /= 2;
        }
        res
    }
    //199
    pub fn right_side_view(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        let mut res = vec![];
        fn search(
            root: Option<&Rc<RefCell<TreeNode>>>,
            current_deep: i32,
            deep: &mut i32,
            res: &mut Vec<i32>,
        ) {
            if let Some(root) = root {
                if current_deep >= *deep {
                    res.push(root.borrow().val);
                    *deep += 1;
                }

                search(root.borrow().right.as_ref(), current_deep + 1, deep, res);
                search(root.borrow().left.as_ref(), current_deep + 1, deep, res);
            }
        }
        let mut deep = 0;
        search(root.as_ref(), 0, &mut deep, &mut res);
        res
    }
    //200
    pub fn num_islands(mut grid: Vec<Vec<char>>) -> i32 {
        fn mark(grid: &mut Vec<Vec<char>>, i: usize, j: usize) {
            grid[i][j] = 'x';
            if i + 1 < grid.len() && grid[i + 1][j] == '1' {
                mark(grid, i + 1, j);
            }
            if j + 1 < grid.first().unwrap().len() && grid[i][j + 1] == '1' {
                mark(grid, i, j + 1);
            }

            if i != 0 && grid[i - 1][j] == '1' {
                mark(grid, i - 1, j);
            }
            if j != 0 && grid[i][j - 1] == '1' {
                mark(grid, i, j - 1);
            }
        }
        let mut res = 0;
        for i in 0..grid.len() {
            for j in 0..grid.first().unwrap().len() {
                if grid[i][j] == '1' {
                    mark(&mut grid, i, j);
                    res += 1;
                }
            }
        }
        res
    }
    //201
    pub fn range_bitwise_and(mut left: i32, mut right: i32) -> i32 {
        let mut shift = 0;
        while left < right {
            left >>= 1;
            right >>= 1;
            shift += 1;
        }
        left << shift
    }
    //202
    pub fn is_happy(n: i32) -> bool {
        fn cal(mut n: i32, res: &mut HashMap<i32, i32>, is_of: &mut bool) {
            let mut r = 0;
            while n != 0 {
                let off = n % 10;
                r += off * off;
                n /= 10;
            }
            if r == 1 {
                *is_of = true;
                return;
            }
            if res.contains_key(&r) {
                *is_of = false;
            } else {
                res.insert(r, 0);
                cal(r, res, is_of);
            }
        }
        let mut res = HashMap::new();
        let mut is_ok = false;
        cal(n, &mut res, &mut is_ok);
        is_ok
    }
    //203
    pub fn remove_elements(mut head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
        fn search_del(node: &mut Option<Box<ListNode>>, val: i32) {
            if node.is_some() {
                loop {
                    let n = node.as_mut().unwrap();
                    if n.val == val {
                        *node = n.next.take();
                    } else {
                        search_del(&mut n.next, val);
                        break;
                    }

                    if node.is_none() || node.as_ref().unwrap().val != val {
                        search_del(node, val);
                        break;
                    }
                }
            }
        }
        search_del(&mut head, val);
        head
    }
    //204
    pub fn count_primes(n: i32) -> i32 {
        let mut is_prime = vec![true; n as usize];
        let mut res = 0;
        for i in 2..n {
            if is_prime[i as usize] {
                res += 1;
                let mut start = 2;
                loop {
                    if start * i < n {
                        is_prime[(start * i) as usize] = false;
                        start += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        res
    }
    //205
    pub fn is_isomorphic(s: String, t: String) -> bool {
        let mut map_s_t = HashMap::new();
        let mut map_t_s = HashMap::new();
        let s = s.as_bytes();
        let t = t.as_bytes();
        for i in 0..s.len() {
            if map_s_t.contains_key(&s[i]) {
                let v = map_s_t.get(&s[i]).unwrap();
                if *v != *&t[i] {
                    return false;
                }
            } else {
                map_s_t.insert(*&s[i], *&t[i]);
            }
            if map_t_s.contains_key(&t[i]) {
                let v = map_t_s.get(&t[i]).unwrap();
                if *v != *&s[i] {
                    return false;
                }
            } else {
                map_t_s.insert(*&t[i], *&s[i]);
            }
        }
        return true;
    }
    // 206
    pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        fn help(
            mut node: Option<Box<ListNode>>,
            next: Option<Box<ListNode>>,
        ) -> Option<Box<ListNode>> {
            if node.is_some() {
                let mut n = node.take();
                let nn = n.as_mut().unwrap().next.take();
                n.as_mut().unwrap().next = next;
                return help(nn, n);
            } else {
                return next;
            }
        }
        return help(head, None);
    }
    //207
    pub fn can_finish(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> bool {
        let mut map = vec![vec![]; num_courses as usize];
        for x in prerequisites {
            let k = *x.first().unwrap() as usize;
            let v = *x.last().unwrap();
            map[k].push(v);
        }

        fn find_cirlce(map: &Vec<Vec<i32>>, prev: &mut Vec<i32>, k: &i32) -> bool {
            if map[*k as usize].len() == 0 {
                return false;
            } else {
                let prereq = map.get(*k as usize).unwrap();
                for i in prereq {
                    if prev.contains(i) {
                        return true;
                    } else {
                        prev.push(*i);
                        let res = find_cirlce(map, prev, i);
                        prev.pop().unwrap();
                        if res {
                            return true;
                        }
                    }
                }
            };
            false
        }

        let mut prev = vec![];
        for i in 0..num_courses {
            prev.push(i);
            if let Some(req) = map.get(i as usize) {
                for next in req {
                    if *next < i {
                        continue;
                    }
                    if find_cirlce(&map, &mut prev, next) {
                        return false;
                    }
                }
            }
            prev.pop().unwrap();
        }
        true
    }
    //208
    fn e_208() {
        struct Trie {
            nodes: Vec<Option<TrieNode>>,
        }
        #[derive(Clone)]
        struct TrieNode {
            charatror: char,
            is_word: bool,
            nexts: Vec<Option<TrieNode>>,
        }
        impl TrieNode {
            fn new(c: char, is_word: bool) -> Self {
                Self {
                    charatror: c,
                    is_word: is_word,
                    nexts: vec![None; 26],
                }
            }
        }
        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl Trie {
            fn new() -> Self {
                Self {
                    nodes: vec![Option::None; 26],
                }
            }

            fn insert(&mut self, word: String) {
                let word = word.as_bytes();
                fn handle_insert(
                    nodes: &mut Vec<Option<TrieNode>>,
                    current_index: usize,
                    word: &[u8],
                ) {
                    if current_index >= word.len() {
                        return;
                    }
                    let word_to_insert = word[current_index];
                    if let Some(node) = nodes.get_mut((word_to_insert - 97) as usize) {
                        if let Some(node) = node {
                            if current_index == word.len() - 1 {
                                node.is_word = true
                            } else {
                                handle_insert(&mut node.nexts, current_index + 1, word);
                            }
                        } else {
                            if current_index == word.len() - 1 {
                                nodes[(word_to_insert - 97) as usize] =
                                    Some(TrieNode::new(word_to_insert as char, true));
                            } else {
                                nodes[(word_to_insert - 97) as usize] =
                                    Some(TrieNode::new(word_to_insert as char, false));
                                handle_insert(
                                    &mut nodes[(word_to_insert - 97) as usize]
                                        .as_mut()
                                        .unwrap()
                                        .nexts,
                                    current_index + 1,
                                    word,
                                );
                            }
                        }
                    }
                }
                handle_insert(&mut self.nodes, 0, word);
            }

            fn search(&self, word: String) -> bool {
                let word = word.as_bytes();
                fn dfs_search(
                    word: &[u8],
                    current_idx: usize,
                    nodes: &Vec<Option<TrieNode>>,
                ) -> bool {
                    if current_idx >= word.len() {
                        return false;
                    }
                    let c_to_search = word[current_idx];
                    let node_idx = (c_to_search - 97) as usize;
                    if let Some(node) = nodes.get(node_idx) {
                        if let Some(node) = node {
                            if node.is_word && current_idx == word.len() - 1 {
                                return true;
                            } else {
                                return dfs_search(word, current_idx + 1, &node.nexts);
                            }
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                return dfs_search(word, 0, &self.nodes);
            }

            fn starts_with(&self, prefix: String) -> bool {
                let word = prefix.as_bytes();
                fn dfs_search(
                    word: &[u8],
                    current_idx: usize,
                    nodes: &Vec<Option<TrieNode>>,
                ) -> bool {
                    if current_idx >= word.len() {
                        return false;
                    }
                    let c_to_search = word[current_idx];
                    let node_idx = (c_to_search - 97) as usize;
                    if let Some(node) = nodes.get(node_idx) {
                        if let Some(node) = node {
                            if current_idx == word.len() - 1 {
                                return true;
                            } else {
                                return dfs_search(word, current_idx + 1, &node.nexts);
                            }
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                return dfs_search(word, 0, &self.nodes);
            }
        }
    }

    //209
    pub fn min_sub_array_len(target: i32, nums: Vec<i32>) -> i32 {
        let mut res = i32::MAX;
        fn search(target: i32, nums: Vec<i32>, res: &mut i32, start: usize) {
            let mut sum = 0;
            for i in start..nums.len() {
                sum += nums[i];
                if sum >= target {
                    if i - start < *res as usize {
                        *res = (i - start + 1) as i32;
                    }
                    search(target, nums, res, start + 1);
                    break;
                }
            }
            if *res == i32::MAX {
                *res = 0;
            }
        }
        search(target, nums, &mut res, 0);
        res
    }
    //210
    pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
        let num_courses = num_courses as usize;
        let mut pre_req = vec![vec![]; num_courses];
        for req in prerequisites {
            pre_req[req[0] as usize].push(req[1]);
        }
        let mut visited = vec![Status::NotVisited; num_courses];

        #[derive(PartialEq, Eq, Clone)]
        enum Status {
            NotVisited,
            Processing,
            Visited,
        }

        fn dfs(
            pre_req: &Vec<Vec<i32>>,
            visited: &mut Vec<Status>,
            res: &mut Vec<i32>,
            course_id: i32,
        ) {
            if visited[course_id as usize] == Status::Visited {
                return;
            }
            let req = pre_req.get(course_id as usize).unwrap();
            for ele in req {
                if visited[*ele as usize] == Status::Visited {
                    continue;
                }
                if visited[*ele as usize] == Status::Processing {
                    res.clear();
                    return;
                }
                visited[*ele as usize] = Status::Processing;
                dfs(pre_req, visited, res, *ele);
                if visited[*ele as usize] == Status::Processing {
                    return;
                }
            }
            visited[course_id as usize] = Status::Visited;
            res.push(course_id);
        }

        let mut res = vec![];
        for i in 0..num_courses {
            if visited[i] == Status::Visited {
                continue;
            }
            visited[i] = Status::Processing;
            dfs(&pre_req, &mut visited, &mut res, i as i32);
            if visited[i] == Status::Processing {
                break;
            }
        }

        res
    }
    //211
    fn solution_211() {
        struct WordDictionary {
            nodes: HashMap<char, DicNode>,
        }

        struct DicNode {
            is_world: bool,
            nexts: HashMap<char, DicNode>,
        }
        impl DicNode {
            fn new(is_world: bool) -> Self {
                Self {
                    is_world,
                    nexts: HashMap::new(),
                }
            }
        }

        impl WordDictionary {
            fn new() -> Self {
                Self {
                    nodes: HashMap::new(),
                }
            }

            fn add_word(&mut self, word: String) {
                let word = word.as_bytes();
                fn dfs_insert(node: &mut HashMap<char, DicNode>, current_idx: usize, word: &[u8]) {
                    let c_to_insert = word[current_idx] as char;
                    let is_last_c = if current_idx == word.len() - 1 {
                        true
                    } else {
                        false
                    };

                    if let Some(next_node) = node.get_mut(&c_to_insert) {
                        if is_last_c {
                            next_node.is_world = true;
                            return;
                        }
                        dfs_insert(&mut next_node.nexts, current_idx + 1, word);
                    } else {
                        let mut dic_node = DicNode::new(if is_last_c { true } else { false });

                        if !is_last_c {
                            dfs_insert(&mut dic_node.nexts, current_idx + 1, word);
                        }
                        node.insert(c_to_insert, dic_node);
                    }
                }
                dfs_insert(&mut self.nodes, 0, word);
            }

            fn search(&self, word: String) -> bool {
                fn dfs_search(
                    node: &HashMap<char, DicNode>,
                    current_idx: usize,
                    word: &[u8],
                ) -> bool {
                    let c_to_search = word[current_idx] as char;

                    if current_idx == word.len() - 1 {
                        if c_to_search == '.' {
                            for (_, v) in node {
                                if v.is_world {
                                    return true;
                                }
                            }
                            return false;
                        } else {
                            if let Some(node) = node.get(&c_to_search) {
                                if node.is_world {
                                    return true;
                                } else {
                                    return false;
                                }
                            } else {
                                return false;
                            }
                        }
                    }

                    if c_to_search == '.' {
                        for (_, v) in node {
                            let res = dfs_search(&v.nexts, current_idx + 1, word);
                            if res {
                                return true;
                            }
                        }
                        return false;
                    } else {
                        if let Some(n) = node.get(&c_to_search) {
                            return dfs_search(&n.nexts, current_idx + 1, word);
                        } else {
                            return false;
                        }
                    }
                }
                dfs_search(&self.nodes, 0, word.as_bytes())
            }
        }
    }

    //216
    pub fn combination_sum3(k: i32, n: i32) -> Vec<Vec<i32>> {
        let mut res = vec![];
        let mut temp = vec![];
        fn search(
            left: usize,
            current: usize,
            temp: &mut Vec<i32>,
            res: &mut Vec<Vec<i32>>,
            n: i32,
        ) -> bool {
            if temp.len() >= left {
                let sum = temp.iter().sum::<i32>();
                if sum == n {
                    res.push(temp.clone());
                    return false;
                } else if sum < n {
                    return true;
                } else {
                    return false;
                }
            }
            for i in current..=8 {
                temp.push((i + 1) as i32);
                let continue_ = search(left, i + 1, temp, res, n);
                temp.pop().unwrap();
                if !continue_ {
                    return true;
                }
            }
            return true;
        }
        search(k as usize, 0, &mut temp, &mut res, n);
        res
    }
    //217
    pub fn contains_duplicate(nums: Vec<i32>) -> bool {
        let mut map = std::collections::HashMap::new();
        for ele in nums.iter() {
            if map.contains_key(ele) {
                return false;
            }
            map.insert(*ele, 0);
        }
        return true;
    }

    //219
    pub fn contains_nearby_duplicate(nums: Vec<i32>, k: i32) -> bool {
        let mut map: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();

        for (idx, v) in nums.iter().enumerate() {
            if let Some(idxs) = map.get_mut(v) {
                for i in idxs.iter() {
                    if i32::abs_diff(idx as i32, *i) <= k as u32 {
                        return true;
                    }
                }
                idxs.push(idx as i32);
            } else {
                map.insert(*v, vec![idx as i32]);
            }
        }
        return false;
    }

    //222
    pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(root: Option<&Rc<RefCell<TreeNode>>>, res: &mut i32) {
            if let Some(root) = root {
                *res += 1;
                dfs(root.borrow().left.as_ref(), res);
                dfs(root.borrow().right.as_ref(), res);
            }
        }
        let mut res = 0;

        dfs(root.as_ref(), &mut res);
        res
    }

    //223
    pub fn compute_area(
        mut ax1: i32,
        mut ay1: i32,
        mut ax2: i32,
        mut ay2: i32,
        mut bx1: i32,
        mut by1: i32,
        mut bx2: i32,
        mut by2: i32,
    ) -> i32 {
        if ax1 > bx1 {
            std::mem::swap(&mut ax1, &mut bx1);
            std::mem::swap(&mut ay1, &mut by1);
            std::mem::swap(&mut ax2, &mut bx2);
            std::mem::swap(&mut ay2, &mut by2);
        };
        let ax = ax2 - ax1;
        let ay = ay2 - ay1;
        let bx = bx2 - bx1;
        let by = by2 - by1;
        if by1 > ay2 || by2 < ay1 || bx1 > ax2 {
            //just return

            return ax * ay + bx * by;
        } else {
            //some overlap
            let max_up = ay2.max(by2);
            let max_down = ay1.min(by1);
            let max_y = max_up - max_down;
            let mut res = 0;
            let max = bx2.max(ax2);
            for x in ax1..max {
                if x < bx1 {
                    res += ay;
                    continue;
                }
                if x >= ax2 {
                    res += by;
                    continue;
                }
                res += max_y;
            }
            return res;
        }
    }

    //225
    fn solution_225() {
        struct MyStack {
            q1: VecDeque<i32>,
            q2: VecDeque<i32>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl MyStack {
            fn new() -> Self {
                Self {
                    q1: VecDeque::new(),
                    q2: VecDeque::new(),
                }
            }

            fn push(&mut self, x: i32) {
                if self.q1.is_empty() {
                    self.q2.push_back(x);
                } else {
                    self.q1.push_back(x);
                }
            }

            fn pop(&mut self) -> i32 {
                let full;
                let empty;
                if self.q1.is_empty() {
                    full = &mut self.q2;
                    empty = &mut self.q1;
                } else {
                    full = &mut self.q1;
                    empty = &mut self.q2;
                };

                for _ in 0..full.len() - 1 {
                    empty.push_back(full.pop_front().unwrap());
                }
                full.pop_front().unwrap()
            }

            fn top(&self) -> i32 {
                if self.q1.is_empty() {
                    *self.q2.iter().last().unwrap()
                } else {
                    *self.q1.iter().last().unwrap()
                }
            }

            fn empty(&self) -> bool {
                self.q1.is_empty() && self.q2.is_empty()
            }
        }
    }

    //226
    pub fn invert_tree(mut root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        if let Some(root) = root.as_mut() {
            let left = Self::invert_tree(root.borrow_mut().left.take());
            let right = Self::invert_tree(root.borrow_mut().right.take());
            root.borrow_mut().left = right;
            root.borrow_mut().right = left;
        }
        return root;
    }

    // 227
    pub fn calculate(s: String) -> i32 {
        let mut back_s = String::new();
        let mut opt_stack = vec![];

        for c in s.chars() {
            let mut number = String::new();
            if c.is_whitespace() {
                continue;
            }
            if c.is_ascii_digit() {
                back_s.push(c);
            } else {
                back_s.push(' ');
                if opt_stack.is_empty() {
                    opt_stack.push(c);
                } else {
                    let last = *opt_stack.last().unwrap();
                    match c {
                        '*' => {
                            if last == '*' || last == '/' {
                                while let Some(x) = opt_stack.pop() {
                                    back_s.push(x);
                                    if let Some(last) = opt_stack.last() {
                                        if *last == '-' || *last == '+' {
                                            break;
                                        }
                                    } else {
                                        break;
                                    }
                                }
                            }
                            opt_stack.push(c);
                        }
                        '/' => {
                            if last == '*' || last == '/' {
                                while let Some(x) = opt_stack.pop() {
                                    back_s.push(x);
                                    if let Some(last) = opt_stack.last() {
                                        if *last == '-' || *last == '+' {
                                            break;
                                        }
                                    } else {
                                        break;
                                    }
                                }
                            }
                            opt_stack.push(c);
                        }
                        '+' => {
                            if last == '*' || last == '/' || last == '-' || last == '+' {
                                while let Some(x) = opt_stack.pop() {
                                    back_s.push(x);
                                    if opt_stack.is_empty() {
                                        break;
                                    }
                                }
                            }
                            opt_stack.push(c);
                        }
                        '-' => {
                            if last == '*' || last == '/' || last == '+' || last == '-' {
                                while let Some(x) = opt_stack.pop() {
                                    back_s.push(x);
                                    if opt_stack.is_empty() {
                                        break;
                                    }
                                }
                            }
                            opt_stack.push(c);
                        }
                        _ => {}
                    }
                }
            }
        }
        while let Some(x) = opt_stack.pop() {
            back_s.push(x);
        }
        println!("{}", back_s);
        let mut stack = vec![];

        let mut number = String::new();
        for ele in back_s.chars() {
            println!("{:?}", stack);
            if ele.is_ascii_digit() {
                number.push(ele);
            } else {
                if !number.is_empty() {
                    stack.push(number.parse::<i32>().unwrap());
                    number.clear();
                }
                if !ele.is_whitespace() {
                    match ele {
                        '+' => {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            stack.push(left + right);
                        }
                        '-' => {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            stack.push(left - right);
                        }
                        '*' => {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            stack.push(left * right);
                        }
                        '/' => {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            stack.push(left / right);
                        }
                        _ => {}
                    }
                }
            }
        }
        if !number.is_empty() {
            stack.push(number.parse::<i32>().unwrap());
            number.clear();
        }
        *stack.last().unwrap()
    }
    //228
    pub fn summary_ranges(nums: Vec<i32>) -> Vec<String> {
        let mut start = 0;
        let mut res = vec![];
        for i in 0..nums.len() - 1 {
            if nums[i] + 1 != nums[i + 1] {
                if i == start {
                    res.push(format!("{}", nums[start]));
                } else {
                    res.push(format!("{}->{}", nums[start], nums[i]));
                }
                start = (i + 1);
            }
        }
        if nums.len() - 1 == start {
            res.push(format!("{}", nums[start]));
        } else {
            res.push(format!("{}->{}", nums[start], *nums.last().unwrap()));
        }
        res
    }
    //229
    pub fn majority_element(nums: Vec<i32>) -> Vec<i32> {
        let times = (nums.len() / 3) as i32;
        let mut map = std::collections::HashMap::new();
        let mut res = vec![];
        for ele in nums {
            if let Some(e) = map.get_mut(&ele) {
                *e += 1;
                if *e == times + 1 {
                    res.push(*e);
                }
            } else {
                map.insert(ele, 1_i32);
            }
        }
        res
    }
    //230
    pub fn kth_smallest(root: Option<Rc<RefCell<TreeNode>>>, k: i32) -> i32 {
        let mut res = vec![];
        fn dfs_search(root: Option<&Rc<RefCell<TreeNode>>>, k: usize, res: &mut Vec<i32>) {
            if let Some(root) = root {
                dfs_search(root.borrow().left.as_ref(), k, res);
                res.push(root.borrow().val);
                if res.len() == k {
                    return;
                }
                dfs_search(root.borrow().right.as_ref(), k, res);
            }
        }
        dfs_search(root.as_ref(), k as usize, &mut res);
        *res.last().unwrap()
    }
    //231
    pub fn is_power_of_two(mut n: i32) -> bool {
        loop {
            if n == 1 {
                return true;
            }
            if n % 2 != 0 {
                return false;
            }
            n /= 2;
        }
    }
    //232
    fn solution_232() {
        struct MyQueue {
            stack1: Vec<i32>,
            stack2: Vec<i32>,
        }
        impl MyQueue {
            fn new() -> Self {
                Self {
                    stack1: vec![],
                    stack2: vec![],
                }
            }

            fn push(&mut self, x: i32) {
                let full;
                if self.stack1.is_empty() {
                    full = &mut self.stack2
                } else {
                    full = &mut self.stack1;
                }
                full.push(x);
            }

            fn pop(&mut self) -> i32 {
                let empty;
                let full;
                if self.stack1.is_empty() {
                    empty = &mut self.stack1;
                    full = &mut self.stack2
                } else {
                    empty = &mut self.stack2;
                    full = &mut self.stack1;
                }
                while let Some(x) = full.pop() {
                    empty.push(x);
                }
                let res = empty.pop().unwrap();
                while let Some(x) = empty.pop() {
                    full.push(x);
                }
                res
            }

            fn peek(&mut self) -> i32 {
                let empty;
                let full;
                if self.stack1.is_empty() {
                    empty = &mut self.stack1;
                    full = &mut self.stack2
                } else {
                    empty = &mut self.stack2;
                    full = &mut self.stack1;
                }
                while let Some(x) = full.pop() {
                    empty.push(x);
                }
                let res = *empty.last().unwrap();
                while let Some(x) = empty.pop() {
                    full.push(x);
                }
                res
            }

            fn empty(&self) -> bool {
                self.stack1.is_empty() && self.stack2.is_empty()
            }
        }
    }
    //233
    pub fn count_digit_one(n: i32) -> i32 {
        if n == 0 {
            return 0;
        }
        let mut n = n as u128;
        let mut dp = vec![vec![1; 10]; 2];
        for i in 1..10 {
            dp[0][i] = 9 * dp[1][i - 1] + 10_u128.pow(i as u32);
            dp[1][i] = dp[0][i] + dp[1][i - 1];
        }
        let mut res = 0;
        fn cal(n: i32, dp: &Vec<u128>, res: &mut i32) {
            if n < 10 {
                if n >= 1 {
                    *res += 1;
                }
            }
            let mut i = 0;
            let mut step = 10;
            while step <= n {
                step *= 10;
                i += 1;
            }
            step /= 10;
            i -= 1;
            *res += dp[i] as i32;
            if n / step >= 2 {
                *res += step;
            } else {
                *res = *res + n - step + 1
            }
            cal(n % step, dp, res);
        }
        cal(n as i32, &dp[1], &mut res);
        res as i32
    }
    // 234
    pub fn is_palindrome_list(head: Option<Box<ListNode>>) -> bool {
        let mut nodes = vec![];
        fn dfs_collect(head: Option<&Box<ListNode>>, nodes: &mut Vec<i32>) {
            if let Some(h) = head {
                nodes.push(h.val);
                dfs_collect(h.next.as_ref(), nodes);
            }
        }
        dfs_collect(head.as_ref(), &mut nodes);
        let mut start = 0 as i32;
        let mut end = (nodes.len() - 1) as i32;
        while start <= end {
            if nodes[start as usize] != nodes[end as usize] {
                return false;
            }
            start += 1;
            end -= 1;
        }
        return true;
    }
    //235
    pub fn lowest_common_ancestor(
        root: Option<Rc<RefCell<TreeNode>>>,
        p: Option<Rc<RefCell<TreeNode>>>,
        q: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        let mut res_a = vec![];
        let mut res_b = vec![];
        fn search(
            root: Option<&Rc<RefCell<TreeNode>>>,
            p: Rc<RefCell<TreeNode>>,
            q: Rc<RefCell<TreeNode>>,
            res_a: &mut Vec<Rc<RefCell<TreeNode>>>,
            res_b: &mut Vec<Rc<RefCell<TreeNode>>>,
        ) -> (bool, bool) {
            if let Some(r) = root {
                let (x1, y1) = search(
                    r.borrow().left.as_ref(),
                    Rc::clone(&p),
                    Rc::clone(&q),
                    res_a,
                    res_b,
                );
                let (x2, y2) = search(
                    r.borrow().right.as_ref(),
                    Rc::clone(&p),
                    Rc::clone(&q),
                    res_a,
                    res_b,
                );
                let self_is_parent_x = r.borrow().val == p.borrow().val;
                let self_is_parent_y = r.borrow().val == q.borrow().val;
                let x = x1 || x2 || self_is_parent_x;
                let y = y1 || y2 || self_is_parent_y;
                if x {
                    res_a.push(Rc::clone(r));
                }
                if y {
                    res_b.push(Rc::clone(r));
                }
                (x, y)
            } else {
                return (false, false);
            }
        }
        search(
            root.as_ref(),
            p.unwrap(),
            q.unwrap(),
            &mut res_a,
            &mut res_b,
        );

        // println!(
        //     "{:?}",
        //     res_a
        //         .iter()
        //         .map(|x| { x.borrow().val })
        //         .collect::<Vec<i32>>()
        // );
        // println!(
        //     "{:?}",
        //     res_b
        //         .iter()
        //         .map(|x| { x.borrow().val })
        //         .collect::<Vec<i32>>()
        // );
        res_a.reverse();
        res_b.reverse();
        let len = res_a.len().min(res_b.len());
        for i in 0..len {
            if res_a[i].borrow().val != res_b[i].borrow().val {
                return Some(Rc::clone(&res_a[i - 1]));
            }
        }
        Some(Rc::clone(&res_a[len - 1]))
    }
    //238
    pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
        let mut res = vec![0; nums.len()];
        let mut right = 1;
        let mut left = 1;
        for i in 1..nums.len() {
            right *= nums[i];
        }
        res[0] = right;
        for i in 1..nums.len() {
            left *= nums[i - 1];
            if i == nums.len() - 1 {
                right = 1;
            } else {
                if nums[i] != 0 {
                    right /= nums[i];
                } else {
                    right = 1;
                    for i in i + 1..nums.len() {
                        right *= nums[i];
                    }
                }
            }
            res[i] = left * right;
        }
        res
    }
    //240
    pub fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
        matrix
            .iter()
            .any(|line| line.binary_search(&target).is_ok())
    }
    //241 bad!!!!!!!!!!!!!!
    pub fn diff_ways_to_compute(expression: String) -> Vec<i32> {
        vec![]
    }

    //242
    pub fn is_anagram(s: String, t: String) -> bool {
        if s.len() != t.len() {
            false
        } else {
            let mut map1 = std::collections::HashMap::new();
            let mut map2 = std::collections::HashMap::new();
            for c in s.chars() {
                if let Some(x) = map1.get_mut(&c) {
                    *x += 1;
                } else {
                    map1.insert(c, 0);
                }
            }
            for c in t.chars() {
                if let Some(x) = map2.get_mut(&c) {
                    *x += 1;
                } else {
                    map2.insert(c, 0);
                }
            }

            map1 == map2
        }
    }
    //257
    pub fn binary_tree_paths(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<String> {
        let mut res = vec![];

        let mut temp = vec![];

        fn dfs_search(
            root: Option<&Rc<RefCell<TreeNode>>>,
            temp: &mut Vec<String>,
            res: &mut Vec<Vec<String>>,
        ) -> bool {
            if let Some(root) = root {
                temp.push(root.borrow().val.to_string());
                let letf = dfs_search(root.borrow().left.as_ref(), temp, res);
                let right = dfs_search(root.borrow().right.as_ref(), temp, res);

                if !letf && !right {
                    if let Some(last) = res.last() {
                        if *last != *temp {
                            res.push(temp.clone());
                        }
                    } else {
                        res.push(temp.clone());
                    }
                }
                temp.pop().unwrap();
                true
            } else {
                false
            }
        }
        dfs_search(root.as_ref(), &mut temp, &mut res);

        res.iter().map(|x| x.join("->")).collect::<Vec<String>>()
    }
    //258
    pub fn add_digits(mut num: i32) -> i32 {
        let mut temp = 0;
        loop {
            while num != 0 {
                temp += num % 10;
                num /= 10;
            }
            num = temp;
            if temp < 10 {
                break;
            }
            temp = 0;
        }
        num
    }
    //260
    pub fn single_number2(nums: Vec<i32>) -> Vec<i32> {
        let mut map = std::collections::HashMap::new();
        for num in nums {
            if map.contains_key(&num) {
                map.remove(&num).unwrap();
            } else {
                map.insert(num, 0);
            }
        }
        map.iter().map(|(k, _)| *k).collect::<Vec<i32>>()
    }
    //263
    pub fn is_ugly(n: i32) -> bool {
        fn find(n: i32, factors: &Vec<i32>) -> bool {
            if n == 1 {
                return true;
            }
            if n == 0 {
                return false;
            }
            for i in factors.iter() {
                if n % *i == 0 {
                    if find(n / *i, factors) {
                        return true;
                    }
                }
            }
            false
        }
        let factors = vec![2, 3, 5];

        return find(n, &factors);
    }
    //264
    pub fn nth_ugly_number(n: i32) -> i32 {
        let factors = vec![2, 3, 5];
        let mut set = std::collections::HashSet::new();
        let mut heap = std::collections::BinaryHeap::new();
        set.insert(1);
        heap.push(std::cmp::Reverse(1));

        for i in 0..n - 1 {
            let next = heap.pop().unwrap().0 as usize;
            for f in factors.iter() {
                if set.insert(*f * next) {
                    heap.push(std::cmp::Reverse(*f * next));
                }
            }
        }
        heap.pop().unwrap().0 as i32
    }
    //268
    pub fn missing_number(nums: Vec<i32>) -> i32 {
        let mut set = std::collections::HashSet::with_capacity(nums.len() + 1);
        for i in 0..=nums.len() {
            set.insert(i as i32);
        }
        for ele in nums {
            set.remove(&ele);
        }
        let e = set.iter().next().unwrap();
        *e
    }

    //274
    pub fn h_index(mut citations: Vec<i32>) -> i32 {
        // citations.sort_by(|a, b| b.cmp(a));
        for (idx, value) in citations.iter().rev().enumerate() {
            if idx + 1 > *value as usize {
                return (idx + 1) as i32;
            }
        }
        citations.len() as i32
    }

    //278
    pub fn first_bad_version(&self, n: i32) -> i32 {
        0
        // let left = 1 as usize;
        // let right = n as usize;
        // while left < right {
        //     let mid = (left + right) / 2;
        //     if self.isBadVersion(mid as i32) {
        //         right = mid - 1;
        //     } else {
        //         left = mid + 1;
        //     }
        //     if right - left <= 5 {
        //         for ii in left..=right {
        //             if self.isBadVersion(ii as i32) {
        //                 return ii as i32;
        //             }
        //         }
        //     }
        // }
        // 1
        // // T T T F F F
        // // 1 2 3 4 5 6
        // // 1 + 6 / 2 = 3
        // // 3 + 6 / 2 = 4
    }
    //279
    pub fn num_squares(n: i32) -> i32 {
        let mut res = i32::MAX;
        for i in 1..res {
            let x = i * i;
            if x > res {
                break;
            }
            if n % x == 0 {
                res = res.min(n / x);
            } else {
                let mut temp = n / x;
                let left = n % x;
                let l = Self::num_squares(left);
                temp += l;
                res = res.min(temp);
            }
        }
        res
    }
    //283
    pub fn move_zeroes(nums: &mut Vec<i32>) {
        let mut slow_ptr = 0;
        let mut fast_ptr = 0;
        while slow_ptr < nums.len() && nums[slow_ptr] != 0 {
            slow_ptr += 1;
        }
        if slow_ptr >= nums.len() - 1 {
            return;
        }
        fast_ptr = slow_ptr + 1;

        loop {
            if fast_ptr != 0 {
                nums[slow_ptr] = nums[fast_ptr];
                nums[fast_ptr] = 0;
                slow_ptr += 1;
            }
            fast_ptr += 1;
            if fast_ptr == nums.len() {
                break;
            }
        }
    }

    //287
    pub fn find_duplicate(nums: Vec<i32>) -> i32 {
        let mut fast = 0;
        let mut slow = 0;
        loop {
            fast = nums[nums[fast] as usize] as usize;
            slow = nums[slow] as usize;
            if fast == slow {
                break;
            }
        }
        slow = 0;
        while slow != fast {
            slow = nums[slow] as usize;
            fast = nums[fast] as usize;
        }
        fast as i32
    }
    //288
    pub fn game_of_life(board: &mut Vec<Vec<i32>>) {
        for i in 0..board.len() {
            for j in 0..board[0].len() {
                let mut live_number = 0;
                // up
                if i >= 1 {
                    let up = board[i - 1][j];
                    if up == 1 || up == 2 || up == 3 {
                        live_number += 1;
                    }
                }
                //down
                if i < board.len() - 1 {
                    let down = board[i + 1][j];
                    if down == 1 || down == 2 || down == 3 {
                        live_number += 1;
                    }
                }
                //left
                if j >= 1 {
                    let left = board[i][j - 1];
                    if left == 1 || left == 2 || left == 3 {
                        live_number += 1;
                    }
                }
                //right
                if j < board[0].len() - 1 {
                    let right = board[i][j + 1];
                    if right == 1 || right == 2 || right == 3 {
                        live_number += 1;
                    }
                }
                //up-left
                if i >= 1 && j >= 1 {
                    let up_left = board[i - 1][j - 1];
                    if up_left == 1 || up_left == 2 || up_left == 3 {
                        live_number += 1;
                    }
                }
                //up-right
                if i >= 1 && j < board[0].len() - 1 {
                    let up_right = board[i - 1][j + 1];
                    if up_right == 1 || up_right == 2 || up_right == 3 {
                        live_number += 1;
                    }
                }
                //down-left
                if i < board.len() - 1 && j >= 1 {
                    let down_left = board[i + 1][j - 1];
                    if down_left == 1 || down_left == 2 || down_left == 3 {
                        live_number += 1;
                    }
                }
                //down-right
                if i < board.len() - 1 && j < board[0].len() - 1 {
                    let down_right = board[i + 1][j + 1];
                    if down_right == 1 || down_right == 2 || down_right == 3 {
                        live_number += 1;
                    }
                }
                let current = board[i][j];
                if current == 1 {
                    match live_number {
                        l if l < 2 => {
                            //die from live to die
                            board[i][j] = 2;
                        }
                        l if l == 2 || l == 3 => {
                            //alive from live to live
                            board[i][j] = 3;
                        }
                        l if l > 3 => {
                            // dir from live to die
                            board[i][j] = 2;
                        }
                        _ => {
                            unreachable!()
                        }
                    }
                } else {
                    match live_number {
                        l if l == 3 => {
                            //alive from die to live
                            board[i][j] = 4;
                        }
                        _ => {
                            //die from die to die
                            board[i][j] = 5;
                            // unreachable!()
                        }
                    }
                }
            }
        }
        for i in 0..board.len() {
            for j in 0..board[0].len() {
                if board[i][j] == 4 || board[i][j] == 3 {
                    board[i][j] = 1;
                } else {
                    board[i][j] = 0;
                }
            }
        }
    }
    //290
    pub fn word_pattern(pattern: String, s: String) -> bool {
        let mut map_c_w = std::collections::HashMap::new();
        let mut map_w_c = std::collections::HashMap::new();
        let chars = pattern.as_bytes();
        let words = s.split_whitespace().collect::<Vec<&str>>();
        if chars.len() != words.len() {
            false
        } else {
            for i in 0..chars.len() {
                if let Some(x) = map_c_w.insert(chars[i], words[i]) {
                    if x != words[i] {
                        return false;
                    }
                }
                if let Some(x) = map_w_c.insert(words[i], chars[i]) {
                    if x != chars[i] {
                        return false;
                    }
                }
            }
            true
        }
    }
    //299
    pub fn get_hint(secret: String, guess: String) -> String {
        let secret = secret.as_bytes();
        let guess = guess.as_bytes();
        let len = secret.len();
        let mut bulls = std::collections::HashSet::new();
        let mut left_sec = std::collections::HashMap::new();
        let mut left_gus = std::collections::HashMap::new();
        for i in 0..len {
            if secret[i] == guess[i] {
                bulls.insert(i);
            }
        }

        for i in 0..len {
            if !bulls.contains(&i) {
                if let Some(x) = left_sec.get_mut(&secret[i]) {
                    *x += 1;
                } else {
                    left_sec.insert(secret[i], 1);
                }
                if let Some(x) = left_gus.get_mut(&guess[i]) {
                    *x += 1;
                } else {
                    left_gus.insert(guess[i], 1);
                }
            }
        }
        let mut cows = 0;
        for (k, v) in left_gus.iter() {
            if let Some(x) = left_sec.get(k) {
                cows += *v.min(x);
            }
        }

        format!("{}A{}B", bulls.len(), cows)
    }
    //play
    pub fn log_play() {
        info!("hello")
    }

    //303
    fn solution_303() {
        struct NumArray {
            nums: Vec<i32>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl NumArray {
            fn new(nums: Vec<i32>) -> Self {
                Self { nums }
            }

            fn sum_range(&self, left: i32, right: i32) -> i32 {
                let left = left as usize;
                let right = right as usize;
                let e = &self.nums[left..=right].iter().sum::<i32>();
                *e
            }
        }
    }
    fn solution_304() {
        struct NumMatrix {
            dp: Vec<Vec<i32>>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl NumMatrix {
            fn new(matrix: Vec<Vec<i32>>) -> Self {
                let mut dp = vec![];
                for line in matrix {
                    let mut l = vec![0];
                    for ele in line {
                        l.push(l.last().unwrap() + ele);
                    }
                    dp.push(l);
                }
                Self { dp }
            }

            fn sum_region(&self, row1: i32, col1: i32, row2: i32, col2: i32) -> i32 {
                let row1 = row1 as usize;
                let col1 = col1 as usize;
                let row2 = row2 as usize;
                let col2 = col2 as usize;
                let mut res = 0;
                for i in row1..=row2 {
                    res += self.dp[i][col2 + 1] - self.dp[i][col1 + 1];
                }
                res
            }
        }
    }
    //306
    pub fn is_additive_number(num: String) -> bool {
        fn is_z_started(n: &str) -> bool {
            n.starts_with('0') && n.len() != 1
        }
        fn search(
            first: (usize, usize),
            sec: (usize, usize),
            max: usize,
            num: &str,
            count: i32,
        ) -> bool {
            if sec.1 == num.len() - 1 {
                return count >= 2;
            }

            let l = &num[first.0..=first.1];
            let r = &num[sec.0..=sec.1];
            if is_z_started(l) {
                return false;
            }
            if is_z_started(r) {
                return false;
            }
            let l: u128 = l.parse().unwrap();
            let r: u128 = r.parse().unwrap();
            for i in 0..2 {
                let from = sec.1 + 1;
                let to = from + max + i;
                if to >= num.len() {
                    return false;
                }
                let evl = &num[from..=to];
                if evl.starts_with('0') && evl.len() != 1 {
                    continue;
                }
                let evl = evl.parse::<u128>().unwrap();
                if l + r == evl {
                    return search(sec, (from, to), to - from, num, count + 1);
                }
            }
            false
        }
        for i in 0..num.len() {
            for j in i..num.len() {
                let from = (0 as usize, i);
                let to = (i + 1, j + 1);
                let max = usize::max(from.1 - from.0, to.1 - to.0);
                if search(from, to, max, &num, 1) {
                    return true;
                }
            }
        }
        false
    }
    //307
    fn solution_307() {
        struct NumArray {
            nums: Vec<i32>,
            dp: Vec<i32>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl NumArray {
            fn new(nums: Vec<i32>) -> Self {
                let mut dp = vec![0];
                for i in nums.iter() {
                    dp.push(dp.last().unwrap() + *i);
                }
                Self { nums, dp }
            }

            fn update(&mut self, index: i32, val: i32) {
                let dif = val - self.nums[index as usize];
                self.nums[index as usize] = val;
                for i in (index + 1) as usize..self.dp.len() {
                    *self.dp.get_mut(i).unwrap() += dif;
                }
            }

            fn sum_range(&self, left: i32, right: i32) -> i32 {
                self.dp[(right + 1) as usize] - self.dp[(left - 1) as usize]
            }
        }
    }

    //310
    pub fn find_min_height_trees(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        let mut map: HashMap<i32, HashSet<i32>> = std::collections::HashMap::new();
        for edge in edges {
            if let Some(x) = map.get_mut(&edge[0]) {
                x.insert(edge[1]);
            } else {
                let mut set = std::collections::HashSet::new();
                set.insert(edge[1]);
                map.insert(edge[0], set);
            }
            if let Some(x) = map.get_mut(&edge[1]) {
                x.insert(edge[0]);
            } else {
                let mut set = std::collections::HashSet::new();
                set.insert(edge[0]);
                map.insert(edge[1], set);
            }
        }
        let mut q = std::collections::VecDeque::new();

        fn search(
            q: &mut VecDeque<Vec<i32>>,
            mut map: HashMap<i32, HashSet<i32>>,
            mut count: i32,
            min: i32,
        ) -> i32 {
            while !q.is_empty() {
                let parents = q.pop_front().unwrap();
                let mut nexts = vec![];
                for p in parents {
                    if let Some(children) = map.remove(&p) {
                        for child in children.iter() {
                            if let Some(x) = map.get_mut(child) {
                                x.remove(&p);
                            }
                        }
                        nexts.append(&mut children.into_iter().collect::<Vec<i32>>());
                    }
                }
                if nexts.len() != 0 {
                    q.push_back(nexts);
                }
                count += 1;
                if count > min {
                    return i32::MAX;
                }
            }
            println!("{:?}", count);
            count
        }
        let mut res = vec![];
        let mut min = i32::MAX;
        for i in 0..n {
            q.push_back(vec![i]);
            let mut count = 0;
            let len = search(&mut q, map.clone(), count, min);
            if len <= min {
                if len < min {
                    res.clear();
                }
                min = len;
                res.push(i);
            }
            q.clear();
        }
        res
    }

    //313
    pub fn nth_super_ugly_number(n: i32, primes: Vec<i32>) -> i32 {
        let mut heap = std::collections::BinaryHeap::new();
        let mut set = std::collections::HashSet::new();
        heap.push(std::cmp::Reverse(1));
        for _ in 0..n - 1 {
            let i = heap.pop().unwrap().0;
            for p in primes.iter() {
                let next = i * *p as usize;
                if set.insert(next) {
                    heap.push(std::cmp::Reverse(next));
                }
            }
        }
        heap.pop().unwrap().0 as i32
    }
    //316
    pub fn remove_duplicate_letters(s: String) -> String {
        let mut map: HashMap<u8, Vec<usize>> = std::collections::HashMap::new();
        let s = s.as_bytes();
        for i in 0..s.len() {
            if let Some(x) = map.get_mut(&s[i]) {
                x.push(i);
            } else {
                map.insert(s[i], vec![i]);
            }
        }
        let mut e = map
            .into_iter()
            .map(|(k, v)| (k, v))
            .collect::<Vec<(u8, Vec<usize>)>>();
        e.sort_by(|x, y| x.0.cmp(&y.0));
        let mut next = 0;
        let mut res = vec![];
        for i in e {
            let mut x = true;
            for k in i.1.iter() {
                if *k >= next {
                    res.push((i.0, *k));
                    next = *k;
                    x = false;
                    break;
                }
            }
            if x {
                res.push((i.0, (*i.1.last().unwrap())));
            }
        }
        println!("{:?}", res);
        "".to_string()
    }
    //318
    pub fn max_product_318(words: Vec<String>) -> i32 {
        let mut sets = Vec::with_capacity(words.len());
        for w in words.iter() {
            let mut x = HashSet::new();
            for xx in w.as_bytes().iter() {
                x.insert(*xx);
            }
            sets.push(x);
        }
        let mut max = 0;
        for i in 0..words.len() - 1 {
            for j in i + 1..words.len() {
                let left = &words[i];
                let right = &words[j];
                let x = &sets[i];
                let y = &sets[j];
                if x.intersection(&y).collect::<Vec<&u8>>().len() != 0 {
                    continue;
                } else {
                    max = max.max(left.len() * right.len())
                }
            }
        }
        max as i32
    }

    //319
    pub fn bulb_switch(n: i32) -> i32 {
        let mut n = n as f32;
        n += 0.5;
        n.sqrt() as i32
    }
    //322
    ///
    /// # what did i lean?
    /// you have to find the optimal structure and in this case it is
    ///
    /// *F(S) = F(S - C) + 1;*
    ///
    /// F(S) the number of coins constructing S --the amount
    ///
    /// if we kown the last one, we can do this.
    ///
    /// but we do not konw the last one , so we have to iterate every possibility.
    pub fn coin_change(mut coins: Vec<i32>, amount: i32) -> i32 {
        fn search(coins: &Vec<i32>, remain: i32, dp: &mut Vec<i32>) -> i32 {
            if remain < 0 {
                return -1;
            }
            if remain == 0 {
                return 0;
            }
            if dp[(remain - 1) as usize] != 0 {
                return dp[(remain - 1) as usize];
            }
            let mut min = i32::MAX;

            for ele in coins {
                let res = search(coins, remain - *ele, dp);
                if res >= 0 && res + 1 < min {
                    min = res + 1;
                }
            }
            dp[(remain - 1) as usize] = if min == i32::MAX { -1 } else { min };
            dp[(remain - 1) as usize]
        }
        let mut dp = vec![0; amount as usize];
        let res = search(&coins, amount, &mut dp);
        res
    }

    //334
    pub fn wiggle_sort(nums: &mut Vec<i32>) {
        nums.sort();
        let mut res = vec![];
        let mut mid = (nums.len() + 1) / 2;

        if nums.len() % 2 != 0 {
            mid -= 1;
            for i in 0..mid {
                res.push(i as i32);
                res.push(nums[i + mid]);
            }
            res.push(nums[nums.len() - mid - 1]);
        } else {
            for i in 0..mid {
                res.push(nums[i + mid]);
                res.push(i as i32);
            }
            res.reverse();
        }
        *nums = res
    }
    //326
    pub fn is_power_of_three(n: i32) -> bool {
        if n == 0 {
            false
        } else {
            let mut n = n;
            loop {
                if n == 1 {
                    return true;
                }
                n /= 3;
                if n % 3 != 0 {
                    break;
                }
            }
            false
        }
    }
    //328
    pub fn odd_even_list(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        if let None = head {
            return head;
        }
        let mut even_head = Box::new(ListNode::new(0));
        fn cal(head: &mut Box<ListNode>, even_head: &mut Box<ListNode>) {
            if let Some(mut next_head) = head.next.take() {
                let next_next_head = next_head.next.take();
                even_head.next = Some(next_head);
                head.next = next_next_head;
                if let Some(next) = head.next.as_mut() {
                    cal(next, even_head.next.as_mut().unwrap());
                }
            }
        }
        fn conjunction(head: &mut Box<ListNode>, last: Option<Box<ListNode>>) {
            if let None = head.next.as_ref() {
                head.next = last
            } else {
                conjunction(head.next.as_mut().unwrap(), last);
            }
        }
        cal(head.as_mut().unwrap(), &mut even_head);
        conjunction(head.as_mut().unwrap(), even_head.next);
        head
    }

    //331
    pub fn is_valid_serialization(preorder: String) -> bool {
        if preorder.starts_with("#") {
            false
        } else {
            let mut stack = vec![];
            let nodes = preorder.split(|c| c == ',').collect::<Vec<&str>>();
            let size = nodes.len() - 1;
            for (idx, node) in nodes.into_iter().enumerate() {
                if let Some(top) = stack.last() {
                    if *top == "#" && node == "#" {
                        loop {
                            stack.pop().unwrap();
                            stack.pop().unwrap();
                            stack.pop().unwrap();
                            if let Some(last) = stack.last() {
                                if *last != "#" {
                                    break;
                                }
                            } else {
                                if idx == size {
                                    return true;
                                }
                                return false;
                            }
                            stack.push(node);
                        }
                        stack.push(node);
                    }
                } else {
                    stack.push(node);
                }
            }
            if stack.len() != 0 { false } else { true }
        }
    }

    //300 😭
    /// # dynamic program..
    /// dp[i] the max in first five..
    /// dp[i] = max(dp[0..i]) + 1;
    pub fn length_of_lis(nums: Vec<i32>) -> i32 {
        let mut dp = vec![1; nums.len()];
        let mut max = 1;
        for i in 1..nums.len() {
            for j in 0..i {
                if nums[j] < nums[i] && dp[i] < dp[j] + 1 {
                    dp[i] = dp[j] + 1;
                    if dp[i] > max {
                        max = dp[i];
                    }
                }
            }
        }
        max
    }
    //334 😭
    pub fn increasing_triplet(nums: Vec<i32>) -> bool {
        if nums.len() < 3 {
            return false;
        } else {
            let mut left_min = vec![nums[0]; nums.len()];
            for i in 1..nums.len() {
                if nums[i] < left_min[i - 1] {
                    left_min[i] = nums[i];
                } else {
                    left_min[i] = left_min[i - 1];
                }
            }

            let mut right_max = vec![nums[nums.len() - 1]; nums.len()];
            for i in (0..nums.len() - 1).rev() {
                if nums[i] > right_max[nums.len() - i - 1] {
                    right_max.push(nums[i]);
                } else {
                    right_max.push(right_max[nums.len() - i - 1]);
                }
            }
            for i in 1..nums.len() - 1 {
                if nums[i] > left_min[i - 1] && nums[i] < right_max[i + 1] {
                    return true;
                }
            }

            false
        }
    }
    //337
    /// 😭
    ///
    /// dp_yes[[p]] = dp_no[[l]] + dp_no[[r]] + p.val
    ///
    /// dp_no[[p]]  = Max(dp_yes[[l]],dp_no[[l]]) + Max(dp_yes[[r]],dp_no[[r]])
    pub fn rob(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(root: Option<&Rc<RefCell<TreeNode>>>) -> (i32, i32) {
            if let Some(root) = root {
                let left = dfs(root.borrow().left.as_ref());
                let right = dfs(root.borrow().right.as_ref());
                let yes = left.1 + right.1 + root.borrow().val;
                let no = left.1.max(left.0) + right.0.max(right.1);
                (yes, no)
            } else {
                (0, 0)
            }
        }
        let e = dfs(root.as_ref());
        e.0.max(e.1)
    }

    // steps
    fn steps(n: i128) -> usize {
        if n <= 2 {
            return 1;
        }
        #[derive(PartialEq, Eq, Hash)]
        enum State {
            OneStep,
            TwoStep,
        }
        fn search(step: i128, stat: State, mem: &mut HashMap<State, Vec<i128>>) -> usize {
            if let Some(m) = mem.get(&stat) {
                if m[step as usize] != -1 {
                    return m[step as usize] as usize;
                }
            }

            let e = if stat == State::OneStep {
                if step == 1 {
                    1
                } else if step == 2 {
                    0
                } else {
                    search(step - 1, State::TwoStep, mem)
                }
            } else {
                if step == 1 {
                    0
                } else if step == 2 {
                    1
                } else {
                    search(step - 2, State::OneStep, mem) + search(step - 2, State::TwoStep, mem)
                }
            };
            let vec = mem.get_mut(&stat).unwrap();
            vec[step as usize] = e as i128;

            e
        }
        let mut mem = HashMap::new();
        mem.insert(State::OneStep, vec![-1; n as usize]);
        mem.insert(State::TwoStep, vec![-1; n as usize]);
        search(n - 2, State::OneStep, &mut mem)
            + search(n - 2, State::TwoStep, &mut mem)
            + search(n - 1, State::TwoStep, &mut mem)
    }

    // 01 knapsack
    pub fn knapsack_(val: Vec<i32>, wei: Vec<i32>, cap: i32) -> i32 {
        let mut dp = vec![vec![0; (cap + 1) as usize]; val.len()];
        for i in 1..val.len() {
            for cap in 1..=cap as usize {
                if wei[i] <= cap as i32 {
                    dp[i][cap] =
                        i32::max(dp[i - 1][cap], dp[i - 1][cap - wei[i] as usize] + val[i]);
                } else {
                    dp[i][cap] = dp[i - 1][cap];
                }
            }
        }
        // dp.iter().for_each(|x| println!("{:?}", x));
        *dp.last().unwrap().last().unwrap()
    }
    pub fn knapscak_back_trace(val: Vec<i32>, wei: Vec<i32>, cap: i32) -> i32 {
        fn search(val: &Vec<i32>, wei: &Vec<i32>, i: usize, cap: i32) -> i32 {
            if i == 0 || cap == 0 {
                0
            } else {
                if wei[i] <= cap {
                    max(
                        search(val, wei, i - 1, cap - wei[i]) + val[i],
                        search(val, wei, i - 1, cap),
                    )
                } else {
                    search(val, wei, i - 1, cap)
                }
            }
        }
        search(&val, &wei, val.len() - 1, cap)
    }
    //338
    pub fn count_bits(n: i32) -> Vec<i32> {
        let mut x = 1;
        let mut res = vec![0, 1];
        for i in 2..=n {
            if i < x * 2 {
                res.push(res[(i - x) as usize] + 1);
            } else if i == x * 2 {
                x *= 2;
                res.push(1);
            }
        }
        res.into_iter().take(n as usize).collect()
    }

    //341
    fn solution_341() {
        #[derive(Debug, PartialEq, Eq)]
        pub enum NestedInteger {
            Int(i32),
            List(Vec<NestedInteger>),
        }
        struct NestedIterator {
            list: Vec<i32>,
            current_idx: usize,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl NestedIterator {
            fn new(nestedList: Vec<NestedInteger>) -> Self {
                fn dfs(nestedList: Vec<NestedInteger>) -> Vec<i32> {
                    let mut r = vec![];
                    for ele in nestedList {
                        match ele {
                            NestedInteger::Int(i) => {
                                r.push(i);
                            }
                            NestedInteger::List(l) => {
                                r.append(&mut dfs(l));
                            }
                        }
                    }
                    r
                }
                let list = dfs(nestedList);
                Self {
                    list,
                    current_idx: 0,
                }
            }

            fn next(&mut self) -> i32 {
                self.current_idx += 1;
                self.list[self.current_idx - 1]
            }

            fn has_next(&self) -> bool {
                self.current_idx < self.list.len()
            }
        }
    }

    //342
    pub fn is_power_of_four(n: i32) -> bool {
        if n == 1 {
            return true;
        }
        let mut n = n;
        loop {
            if n % 4 != 0 {
                return false;
            }
            n /= 4;
            if n == 1 {
                return true;
            }
        }
    }
    //343
    pub fn integer_break(n: i32) -> i32 {
        let n = (n / 3) as u32;
        let left = n % 3;
        if left == 0 {
            3_i32.pow(n)
        } else if left == 2 {
            3_i32.pow(n) * 2
        } else {
            3_i32.pow(n - 1) * 4
        }
    }
    //344
    pub fn reverse_string(s: &mut Vec<char>) {
        let mid = s.len();
        let len = s.len() - 1;
        for i in 0..mid {
            let t = s[i];
            s[i] = s[len - i];
            s[len - i] = t;
            println!("{}->{}", s[i], s[len - i]);
        }
    }
    //345
    pub fn reverse_vowels(mut s: String) -> String {
        let mut set = std::collections::HashSet::new();
        set.insert('e');
        set.insert('E');
        set.insert('o');
        set.insert('O');
        set.insert('u');
        set.insert('U');
        set.insert('i');
        set.insert('I');
        set.insert('a');
        set.insert('A');

        unsafe {
            let ss = s.as_bytes_mut();
            let mut left = 0;
            let mut right = ss.len() - 1;
            while left < right {
                let l = ss[left];
                let r = ss[right];
                if !set.contains(&(l as char)) {
                    left += 1;
                    continue;
                }
                if !set.contains(&(r as char)) {
                    right -= 1;
                    continue;
                }
                ss[left] = r;
                ss[right] = l;
                left += 1;
                right -= 1;
            }
            s
        }
    }
    //347
    pub fn top_k_frequent(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut map = std::collections::HashMap::new();
        for ele in nums {
            if let Some(e) = map.get_mut(&ele) {
                *e += 1;
            } else {
                map.insert(ele, 0);
            }
        }
        let mut e = map.into_iter().map(|x| x).collect::<Vec<(i32, i32)>>();
        e.sort_by(|a, b| b.1.cmp(&a.1));
        e.into_iter()
            .take(k as usize)
            .map(|(k, v)| k)
            .collect::<Vec<i32>>()
    }
    //349
    pub fn intersection(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        let num1: std::collections::HashSet<i32, std::hash::RandomState> =
            std::collections::HashSet::from_iter(nums1.into_iter());
        let num2: std::collections::HashSet<i32, std::hash::RandomState> =
            std::collections::HashSet::from_iter(nums2.into_iter());

        num1.intersection(&num2).map(|x| *x).collect::<Vec<i32>>()
    }

    //350

    pub fn intersect(nums1: Vec<i32>, nums2: Vec<i32>) -> Vec<i32> {
        let max;
        let min;
        let mut res = vec![];
        if nums1.len() > nums2.len() {
            max = nums1;
            min = nums2;
        } else {
            min = nums2;
            max = nums1;
        }

        let mut max_map = std::collections::HashMap::new();
        let mut min_map = std::collections::HashMap::new();

        for ele in max {
            if let Some(x) = max_map.get_mut(&ele) {
                *x += 1;
            } else {
                max_map.insert(ele, 1);
            }
        }

        for ele in min {
            if let Some(x) = min_map.get_mut(&ele) {
                *x += 1;
            } else {
                min_map.insert(ele, 1);
            }
        }
        for (k, v) in min_map {
            if let Some(x) = max_map.get(&k) {
                res.append(&mut [k].repeat(v.min(*x) as usize));
            }
        }
        res
    }
    //355
    fn solution_355() {
        struct Twitter {
            tweets: Vec<(i32, i32)>,
            followers: std::collections::HashMap<i32, std::collections::HashSet<i32>>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl Twitter {
            fn new() -> Self {
                Self {
                    tweets: vec![],
                    followers: std::collections::HashMap::new(),
                }
            }

            fn post_tweet(&mut self, user_id: i32, tweet_id: i32) {
                self.tweets.push((user_id, tweet_id));
            }

            fn get_news_feed(&self, user_id: i32) -> Vec<i32> {
                let mut res = vec![];
                if let Some(followers) = self.followers.get(&user_id) {
                    for (user_id_, tweet_id) in self.tweets.iter().rev() {
                        if *user_id_ == user_id || followers.contains(user_id_) {
                            res.push(*tweet_id);
                        }
                        if res.len() == 10 {
                            break;
                        }
                    }
                } else {
                    for (user_id_, tweet_id) in self.tweets.iter().rev() {
                        if *user_id_ == user_id {
                            res.push(*tweet_id);
                        }
                        if res.len() == 10 {
                            break;
                        }
                    }
                }
                res
            }

            fn follow(&mut self, follower_id: i32, followee_id: i32) {
                if let Some(f) = self.followers.get_mut(&follower_id) {
                    f.insert(followee_id);
                } else {
                    let mut f = std::collections::HashSet::new();
                    f.insert(followee_id);
                    self.followers.insert(follower_id, f);
                }
            }

            fn unfollow(&mut self, follower_id: i32, followee_id: i32) {
                if let Some(f) = self.followers.get_mut(&follower_id) {
                    f.remove(&followee_id);
                }
            }
        }
    }

    //357
    pub fn count_numbers_with_unique_digits(n: i32) -> i32 {
        let mut dp_2 = vec![9 as i32; n as usize];
        for i in 1..n as usize {
            dp_2[i] = dp_2[i - 1] * (10 - i as i32);
        }
        let mut dp = vec![1; (n + 1) as usize];
        for i in 1..=n as usize {
            dp[i] = dp[i - 1] + dp_2[i - 1];
        }
        dp[n as usize]
    }

    //365
    pub fn can_measure_water(x: i32, y: i32, target: i32) -> bool {
        let mut set = std::collections::HashSet::new();
        fn dfs(
            current: (i32, i32),
            res: &mut std::collections::HashSet<(i32, i32)>,
            max: (i32, i32),
            r: &mut bool,
            target: i32,
        ) {
            if *r || res.contains(&current) {
                return;
            }
            if current.0 + current.1 == target {
                *r = true;
                return;
            }

            res.insert(current);
            // x = 0 , y != 0
            if current.0 == 0 && current.1 != 0 {
                // full x
                dfs((max.0, current.1), res, max, r, target);
                // y -> x
                let next;
                if current.1 >= max.0 {
                    next = (max.0, current.1 - max.0);
                } else {
                    next = (current.1, 0);
                }
                dfs(next, res, max, r, target);
                //x = 0, y = 0
            } else if current.0 == 0 && current.1 == 0 {
                // full x
                dfs((max.0, 0), res, max, r, target);
                // full y
                dfs((0, max.1), res, max, r, target);
            } else if current.0 != 0 && current.1 == 0 {
                // full y
                dfs((current.0, max.1), res, max, r, target);
                // x-> y
                let next;
                if current.0 >= max.1 {
                    next = (current.0 - max.1, max.1);
                } else {
                    next = (0, current.0);
                }
                dfs(next, res, max, r, target);
            } else if current.0 != 0 && current.1 != 0 {
                //full x
                dfs((max.0, current.1), res, max, r, target);
                //full y
                dfs((current.0, max.1), res, max, r, target);
                // empty x
                dfs((0, current.1), res, max, r, target);
                //empty y
                dfs((current.0, 0), res, max, r, target);
                //x -> y
                let y_left = max.1 - current.1;
                if y_left >= current.0 {
                    dfs((0, current.1 + current.0), res, max, r, target);
                } else {
                    dfs((current.0 - y_left, max.1), res, max, r, target);
                }
                //y -> x
                let x_left = max.0 - current.0;
                if current.1 <= x_left {
                    dfs((current.0 + current.1, 0), res, max, r, target);
                } else {
                    dfs((max.0, current.1 - x_left), res, max, r, target);
                }
            }
        }
        let mut r = false;
        dfs((0, 0), &mut set, (x, y), &mut r, target);
        r
    }
    //367
    pub fn is_perfect_square(num: i32) -> bool {
        if num == 1 {
            return true;
        }
        let mut left = 2;
        let mut right = num / 2;
        while left <= right {
            let middle = (left + right) / 2;
            let x = (middle as usize) * (middle as usize);
            if x > num as usize {
                right = middle - 1;
            } else if x < num as usize {
                left = middle + 1;
            } else {
                return true;
            }
        }
        false
    }
    //368
    pub fn largest_divisible_subset(nums: Vec<i32>) -> Vec<i32> {
        let mut nums = nums;
        nums.sort();
        let mut dp = vec![(1, 0); nums.len()];
        for e in dp.iter_mut().enumerate() {
            e.1.1 = e.0;
        }
        let mut max = 1;
        let mut max_idx = 0;
        for i in 1..nums.len() {
            for j in (0..i - 1).rev() {
                if nums[i] % nums[j] == 0 {
                    dp[i] = (dp[j].0 + 1, j);
                    if dp[i].0 > max {
                        max = dp[i].0;
                        max_idx = i;
                    }
                }
            }
        }
        let mut res = vec![];
        loop {
            res.push(nums[max_idx]);
            if max_idx == dp[max_idx].1 {
                break;
            }
            max_idx = dp[max_idx].1;
        }
        res
    }

    // //371
    // pub fn get_sum(a: i32, b: i32) -> i32 {

    // }

    //372
    pub fn super_pow(mut a: i32, b: Vec<i32>) -> i32 {
        fn pow_wlth_mod(a: i32, n: i32, m: i32) -> i32 {
            let a = a as u128;
            let m = m as u128;
            let mut r: u128 = 1;
            for i in 0..n {
                r = (r * a) % m;
            }
            r as i32
        }
        let m = 1337;
        let mut r = 1;
        for i in (0..b.len()).rev() {
            r = (r * pow_wlth_mod(a, b[i], m)) % m;
            a = pow_wlth_mod(a, 10, m);
        }
        r
    }

    //373
    pub fn k_smallest_pairs(nums1: Vec<i32>, nums2: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
        // let mut res = vec![];
        // let mut kk = vec![];
        // for i in 0..nums1.len() {
        //     kk.push((i, 0, nums1[i] + nums2[0]));
        // }
        // while res.len() != k as usize {
        //     let mut min = *kk.iter().min_by(|x, y| x.2.cmp(&y.2)).unwrap();
        //     res.push(vec![nums1[min.0], nums2[min.1]]);
        //     if min.1 + 1 >= nums2.len()  {
        //         min = (min.0, min.1 + 1, i32::MAX);
        //     } else {
        //         min = (min.0, min.1 + 1, nums1[min.0] + nums2[min.1 + 1]);
        //     }
        //     kk[min.0] = min;
        // }
        // res
        struct Item(usize, usize, i32);
        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(std::cmp::Reverse(self.2).cmp(&std::cmp::Reverse(other.2)))
            }
        }

        impl PartialEq for Item {
            fn eq(&self, other: &Self) -> bool {
                self.2 == other.2
            }
        }
        impl Eq for Item {}
        impl Ord for Item {
            fn cmp(&self, other: &Self) -> Ordering {
                // std::cmp::Reverse(self.2).cmp(&std::cmp::Reverse(other.2))
                self.partial_cmp(other).unwrap()
            }
        }
        let mut res = vec![];

        let mut heap = std::collections::BinaryHeap::new();
        for i in 0..nums1.len() {
            heap.push(Item(i, 0, nums1[i] + nums2[0]));
        }
        while res.len() != k as usize {
            let mut next = heap.pop().unwrap();
            res.push(vec![nums1[next.0], nums2[next.1]]);
            if next.1 + 1 >= nums2.len() {
                next.2 = i32::MAX;
            } else {
                next.1 = next.1 + 1;
                next.2 = nums1[next.0] + nums2[next.1];
            }
            heap.push(next);
        }
        res
    }
    //374
    unsafe fn guessNumber(n: i32) -> i32 {
        fn guess(num: i32) -> i32 {
            0
        }
        let mut left = 1 as u128;
        let mut right = n as u128;
        loop {
            let mid = (left + right) / 2;
            let r = guess(mid as i32);
            if r == 0 {
                return n;
            } else if r == -1 {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }
    //             (i <= k <= j)
    //375 dp(i,j) = min(k + dp(i,k - 1),dp(k + 1,j));
    pub fn get_money_amount(n: i32) -> i32 {
        let mut dp = vec![vec![0; (n + 2) as usize]; (n + 2) as usize];
        for i in 1..n as usize {
            for j in 1..=n as usize - i {
                dp[j][j + i] = i32::MAX;
                for k in j..=j + i {
                    let max = dp[j][k - 1].max(dp[k + 1][j + i]);
                    if max as usize + k < dp[j][j + i] as usize {
                        dp[j][j + i] = max + k as i32;
                    }
                }
            }
        }
        // println!("{:?}",dp);
        dp[1][n as usize]
    }

    //376
    pub fn wiggle_max_length(nums: Vec<i32>) -> i32 {
        let mut x = vec![];
        for i in 1..nums.len() {
            if nums[i] > nums[i - 1] {
                x.push(1);
            } else if nums[i] < nums[i - 1] {
                x.push(0);
            }
        }
        println!("x{:?}", x);

        let mut dp = vec![1; x.len()];
        for i in 1..dp.len() {
            for j in (0..=i - 1).rev() {
                if x[j] != x[i] {
                    dp[i] += dp[j];
                    break;
                }
            }
        }
        println!("x{:?}", dp);
        if let Some(last) = dp.last() {
            *last + 1
        } else {
            1
        }
    }

    //377
    pub fn combination_sum4(nums: Vec<i32>, target: i32) -> i32 {
        // let mut nums = nums;
        // let mut res = 0;
        // nums.sort();
        // fn dfs(res: &mut i32, nums: &Vec<i32>, target: i32, current: i32) {
        //     for num in nums {
        //         let c = current + *num;
        //         if c > target {
        //             break;
        //         }
        //         if c == target {
        //             *res += 1;
        //             break;
        //         }
        //         dfs(res, nums, target, c);
        //     }
        // }
        // dfs(&mut res, &nums, target, 0);
        // res
        let mut dp = vec![0; (target + 1) as usize];
        dp[0] = 1;
        for i in 0..dp.len() {
            for ele in nums.iter() {
                if *ele as usize <= i {
                    dp[i] += dp[i - *ele as usize]
                }
            }
        }
        *dp.last().unwrap()
    }
    //378
    pub fn kth_smallest_2(matrix: Vec<Vec<i32>>, k: i32) -> i32 {
        let mut nums = Vec::with_capacity(matrix.len() * matrix[0].len());
        for mut m in matrix {
            nums.append(&mut m);
        }
        nums.sort();
        nums[k as usize]
    }
    //380
    pub fn solution_380() {
        use std::collections;
        struct RandomizedSet {
            set: HashSet<i32>,
            list: VecDeque<i32>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl RandomizedSet {
            fn new() -> Self {
                Self {
                    set: HashSet::new(),
                    list: VecDeque::new(),
                }
            }

            fn insert(&mut self, val: i32) -> bool {
                self.list.push_back(val);
                self.set.insert(val)
            }

            fn remove(&mut self, val: i32) -> bool {
                self.set.remove(&val)
            }

            fn get_random(&mut self) -> i32 {
                use rand::Rng;

                self.list = self
                    .list
                    .iter()
                    .filter(|x| self.set.contains(x))
                    .map(|x| *x)
                    .collect::<VecDeque<i32>>();
                let mut rng = rand::thread_rng();
                let idx = rng.gen_range(0..(self.list.len()));
                self.list[idx]
            }
        }
    }
    //382
    fn solution_382() {
        struct Solution {
            list: VecDeque<i32>,
        }
        impl Solution {
            fn new(mut head: Option<Box<ListNode>>) -> Self {
                let mut list = VecDeque::new();
                loop {
                    if head.is_none() {
                        break;
                    } else {
                        let mut h = head.unwrap();
                        list.push_back(h.val);
                        head = h.next.take();
                    }
                }
                Self { list }
            }

            fn get_random(&mut self) -> i32 {
                let next = self.list.pop_back().unwrap();
                self.list.push_front(next);
                next
            }
        }
    }

    //383
    pub fn can_construct(ransom_note: String, magazine: String) -> bool {
        use std::collections::HashMap;
        let mut map1 = HashMap::new();
        let mut map2 = HashMap::new();
        for c in ransom_note.chars() {
            if let Some(e) = map1.get_mut(&c) {
                *e += 1;
            } else {
                map1.insert(c, 1);
            }
        }

        for c in magazine.chars() {
            if let Some(e) = map2.get_mut(&c) {
                *e += 1;
            } else {
                map2.insert(c, 1);
            }
        }
        for (k, v) in map1 {
            if let Some(v2) = map2.get(&k) {
                if *v2 < v {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    //384
    pub fn solution_384() {
        struct Solution {
            list: Vec<i32>,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl Solution {
            fn new(nums: Vec<i32>) -> Self {
                Self { list: nums }
            }

            fn reset(&self) -> Vec<i32> {
                self.list.clone()
            }

            fn shuffle(&self) -> Vec<i32> {
                use rand;
                use rand::Rng;
                let mut list = self.list.clone();
                let mut rng = rand::thread_rng();
                let mut res = vec![];
                while res.len() < self.list.len() {
                    let current_len = res.len();
                    let next_idx = rng.gen_range(0..self.list.len() - current_len);
                    let next = list[next_idx];
                    list[next_idx] = *list.last().unwrap();
                    list.pop();
                    res.push(next);
                }
                res
            }
        }
    }

    //385

    pub fn deserialize(s: String) -> NestedInteger {
        fn dfs(nested: &mut NestedInteger, mut start_idx: usize, s: &[u8]) -> usize {
            match nested {
                NestedInteger::List(l) => {
                    let mut cache = String::new();
                    while start_idx < s.len() {
                        let c = s[start_idx] as char;
                        if c.is_numeric() || c == '-' {
                            cache.push(c);
                        } else if c == ',' {
                            if cache.len() != 0 {
                                let next = cache.parse::<i32>().expect("convert fails");
                                l.push(NestedInteger::Int(next));
                                cache.clear();
                            }
                        } else if c == ']' {
                            if cache.len() != 0 {
                                let next = cache.parse::<i32>().expect("convert fails");
                                l.push(NestedInteger::Int(next));
                            }
                            return start_idx + 1;
                        } else if c == '[' {
                            let mut new_list = NestedInteger::List(vec![]);
                            start_idx = dfs(&mut new_list, start_idx + 1, s);
                            l.push(new_list);
                            continue;
                        }
                        start_idx += 1;
                    }
                    start_idx
                }
                _ => {
                    unreachable!()
                }
            }
        }

        if s.starts_with('[') {
            let mut res = NestedInteger::List(vec![]);
            dfs(&mut res, 1, s.as_bytes());
            res
        } else {
            NestedInteger::Int(s.parse::<i32>().unwrap())
        }
    }
    //386
    pub fn lexical_order(n: i32) -> Vec<i32> {
        let mut x = 1;
        (0..n as usize).fold(vec![0; n as usize], |mut res, i| {
            res[i] = x;
            x *= 10;
            if x > n {
                while x % 10 == 9 || x + 1 > n {
                    x /= 10;
                }
                x += 1;
            }
            res
        })
    }

    //387
    pub fn first_uniq_char(s: String) -> i32 {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        for ele in s.as_bytes() {
            if let Some(v) = map.get_mut(ele) {
                *v += 1;
            } else {
                map.insert(*ele, 1);
            }
        }
        for (idx, ele) in s.as_bytes().iter().enumerate() {
            if let Some(v) = map.get(&ele) {
                if *v == 1 {
                    return idx as i32;
                }
            }
        }
        -1
    }
    //388
    pub fn length_longest_path(input: String) -> i32 {
        let mut stack = vec![];
        let mut cache = String::new();
        let mut i = 0;
        let mut res = 0;
        let input = input.as_bytes();
        let mut expect = 2;
        let mut real = 1;
        let mut in_world = false;
        while i < input.len() {
            let c = input[i] as char;

            if c.is_alphanumeric() || c == '.' || c == ' ' {
                if c == ' ' && in_world {
                    cache.push(c);
                } else {
                    cache.push(c);
                }
                in_world = true;
            } else if c == '\n' {
                in_world = false;
                // println!("{:?}", cache);
                loop {
                    if input[i + 1] == b'\t' || input[i + 1] == b' ' {
                        if input[i + 1] == b' ' {
                            i += 1;
                            continue;
                        }
                        i += 1;
                        real += 1;
                    } else {
                        break;
                    }
                }
                stack.push(cache.clone());
                if cache.contains('.') {
                    println!("{:?}", stack.join("/"));
                    let len = stack.join("/").len();
                    if len > res {
                        res = len;
                    }
                }

                cache.clear();

                if expect > real {
                    let pop_num = (expect - real) as usize;
                    for _ in 0..pop_num {
                        stack.pop();
                    }
                    expect = real + 1;
                } else if real > expect {
                    expect = real + 1;
                } else {
                    expect += 1;
                }
                real = 1;
            }

            i += 1;
        }
        if cache.contains('.') {
            println!("{}", "eeeeeee");
            stack.push(cache);
            println!("{:?}", stack.join("/"));
            let len = stack.join("/").len();
            if len > res {
                res = len;
            }
        }
        res as i32
    }

    //389
    pub fn find_the_difference(s: String, t: String) -> char {
        use std::collections::HashMap;
        let mut map1 = HashMap::new();
        let mut map2 = HashMap::new();
        for c in s.chars() {
            if let Some(v) = map1.get_mut(&c) {
                *v += 1;
            } else {
                map1.insert(c, 1);
            }
        }
        for c in t.chars() {
            if let Some(v) = map2.get_mut(&c) {
                *v += 1;
            } else {
                map2.insert(c, 1);
            }
        }
        for (k, v) in map2.into_iter() {
            if let Some(x) = map1.remove(&k) {
                if x != v {
                    return k;
                }
            } else {
                return k;
            }
        }
        unreachable!();
    }
    //390
    pub fn last_remaining(n: i32) -> i32 {
        fn search(a1: i32, n: i32, step: i32, k: i32) -> i32 {
            if n == 1 {
                a1
            } else {
                match k % 2 {
                    0 => match n % 2 {
                        0 => search(a1 + step, n / 2, step * 2, k + 1),
                        _ => search(a1 + step, n / 2, step * 2, k + 1),
                    },
                    _ => match n % 2 {
                        0 => search(a1, n / 2, step * 2, k + 1),
                        _ => search(a1 + step, n / 2, step * 2, k + 1),
                    },
                }
            }
        }
        search(1, n, 1, 0)
    }
    //393
    pub fn valid_utf8(data: Vec<i32>) -> bool {
        fn is_followed(d: i32) -> bool {
            d >= 128 && d < 192
        }
        fn is_one(d: i32) -> bool {
            d < 128
        }
        fn is_two(d: i32) -> bool {
            d >= 192 && d < 224
        }
        fn is_three(d: i32) -> bool {
            d >= 224 && d < 240
        }
        fn is_four(d: i32) -> bool {
            d >= 240 && d < 248
        }
        #[derive(PartialEq, Eq)]
        enum Type {
            Fore,
            Three,
            Two,
            One,
            Follow,
            Invalid,
        }
        fn parse(d: i32) -> Type {
            if is_followed(d) {
                Type::Follow
            } else if is_four(d) {
                Type::Fore
            } else if is_three(d) {
                Type::Three
            } else if is_two(d) {
                Type::Two
            } else if is_one(d) {
                Type::One
            } else {
                Type::Invalid
            }
        }
        let mut idx = 0;
        while idx < data.len() {
            let ele = data[idx];
            match parse(ele) {
                Type::Follow => {
                    return false;
                }
                Type::Fore => {
                    for _ in 0..3 {
                        idx += 1;
                        if idx >= data.len() {
                            return false;
                        }
                        if parse(data[idx]) != Type::Follow {
                            return false;
                        }
                    }
                }
                Type::Three => {
                    for _ in 0..2 {
                        idx += 1;
                        if idx >= data.len() {
                            return false;
                        }
                        if parse(data[idx]) != Type::Follow {
                            return false;
                        }
                    }
                }
                Type::Two => {
                    for _ in 0..1 {
                        idx += 1;
                        if idx >= data.len() {
                            return false;
                        }
                        if parse(data[idx]) != Type::Follow {
                            return false;
                        }
                    }
                }
                Type::One => {}
                Type::Invalid => {
                    return false;
                }
            }
            idx += 1;
        }
        true
    }

    //394
    pub fn decode_string(s: String) -> String {
        fn decode(s: &[u8], mut idx: usize, r: &mut String) -> usize {
            let mut repeat_times = String::new();
            while idx < s.len() {
                let c = s[idx] as char;
                if c == ']' {
                    return idx;
                } else if c.is_alphabetic() {
                    r.push(c);
                } else if c.is_numeric() {
                    repeat_times.push(c);
                } else if c == '[' {
                    let times = repeat_times.parse::<usize>().unwrap();
                    let mut res = String::new();
                    idx = decode(s, idx + 1, &mut res);
                    r.push_str(&res.repeat(times));
                    repeat_times.clear();
                }
                idx += 1;
            }
            unreachable!()
        }
        let s = s.as_bytes();
        let mut idx = 0;
        let mut res = String::new();
        let mut nums = String::new();
        while idx < s.len() {
            let c = s[idx] as char;
            if c.is_alphabetic() {
                res.push(c);
            } else if c.is_numeric() {
                nums.push(c);
            } else if c == '[' {
                let repeat_times = nums.parse::<usize>().unwrap();
                let mut r = String::new();
                let i = decode(s, idx + 1, &mut r);
                res.push_str(&r.repeat(repeat_times));
                idx = i;
                nums.clear();
            }
            idx += 1;
        }
        res
    }
    //395 divide and conqur
    pub fn longest_substring(s: String, k: i32) -> i32 {
        use std::collections::HashMap;
        let mut res = 0;
        fn search(s: &[u8], k: i32, r: &mut usize) {
            if s.len() < k as usize {
                return;
            }
            let mut map = HashMap::new();
            for i in 0..s.len() {
                if let Some(v) = map.get_mut(&s[i]) {
                    *v += 1;
                } else {
                    map.insert(s[i], 1);
                }
            }
            let map = map
                .into_iter()
                .filter(|(_, v)| *v < k)
                .collect::<HashMap<u8, i32>>();
            let res = s.split(|x| map.contains_key(x)).collect::<Vec<&[u8]>>();
            if res.len() == 1 {
                if *r < res[0].len() {
                    *r = res[0].len();
                }
            } else {
                for e in res {
                    search(e, k, r);
                }
            }
        }
        search(s.as_bytes(), k, &mut res);
        res as i32
    }
    //369
    pub fn max_rotate_function(nums: Vec<i32>) -> i32 {
        let sum = nums.iter().sum::<i32>();

        let mut res = 0;
        fn calc(nums: &Vec<i32>, n: usize, sum: i32, max: &mut i32) -> i32 {
            if n == 0 {
                let mut r = 0;
                for i in 0..nums.len() {
                    r += i as i32 * nums[i];
                }

                *max = r.max(*max);
                r
            } else {
                let r =
                    calc(nums, n - 1, sum, max) + (nums.len() as i32 + 1) * nums[nums.len() - n];
                *max = r.max(*max);
                r
            }
        }
        let r = calc(&nums, nums.len() - 1, sum, &mut res);
        res = res.max(r);
        res
    }

    //397
    pub fn integer_replacement(n: i32) -> i32 {
        use std::collections::HashMap;
        let mut dp = HashMap::new();
        fn search(dp: &mut HashMap<usize, i32>, n: usize) -> i32 {
            if n == 1 {
                return 0;
            }
            if let Some(v) = dp.get(&n) {
                return *v;
            }
            if n % 2 == 0 {
                let res = search(dp, n / 2) + 1;

                dp.insert(n, res);
                res
            } else {
                if !dp.contains_key(&(n + 1)) {
                    search(dp, n + 1);
                }
                if !dp.contains_key(&(n - 1)) {
                    search(dp, n - 1);
                }
                let min = *dp.get(&(n + 1)).unwrap().min(dp.get(&(n - 1)).unwrap()) + 1;
                dp.insert(n, min);
                min
            }
        }

        search(&mut dp, n as usize)
    }

    //398
    fn solution_398() {
        use rand::rngs::ThreadRng;
        use std::collections::HashMap;
        struct Solution {
            map: HashMap<i32, Vec<usize>>,
            r: ThreadRng,
        }

        /**
         * `&self` means the method takes an immutable reference.
         * If you need a mutable reference, change it to `&mut self` instead.
         */
        impl Solution {
            fn new(nums: Vec<i32>) -> Self {
                let mut map: HashMap<i32, Vec<usize>> = HashMap::new();
                for (idx, v) in nums.into_iter().enumerate() {
                    if let Some(v) = map.get_mut(&v) {
                        v.push(idx);
                    } else {
                        map.insert(v, vec![idx]);
                    }
                }
                let r = rand::thread_rng();
                Self { map, r }
            }

            fn pick(&mut self, target: i32) -> i32 {
                if let Some(vec) = self.map.get(&target) {
                    let e = self.r.gen_range(0..vec.len());
                    return vec[e] as i32;
                }
                unreachable!()
            }
        }
    }

    //399

    pub fn calc_equation(
        equations: Vec<Vec<String>>,
        values: Vec<f64>,
        queries: Vec<Vec<String>>,
    ) -> Vec<f64> {
        use std::collections::{HashMap, HashSet};
        let mut map: HashMap<String, Vec<(String, f64)>> = HashMap::new();
        for (idx, eq) in equations.into_iter().enumerate() {
            if let Some(vals) = map.get_mut(&eq[0]) {
                vals.push((eq[1].clone(), values[idx]));
            } else {
                map.insert(eq[0].clone(), vec![(eq[1].clone(), values[idx])]);
            }

            if let Some(vals) = map.get_mut(&eq[1]) {
                vals.push((eq[0].clone(), 1.0 / values[idx]));
            } else {
                map.insert(eq[1].clone(), vec![(eq[0].clone(), 1.0 / values[idx])]);
            }
        }
        println!("{:?}", map);
        let mut res = vec![];

        fn search(
            current: f64,
            query: &String,
            final_: &String,
            map: &HashMap<String, Vec<(String, f64)>>,
            set: &mut HashSet<String>,
        ) -> f64 {
            if let Some(e) = map.get(query) {
                for (right, value) in e {
                    if set.contains(right) {
                        continue;
                    }
                    // search(current * *value, right, final_, map);
                    if right == final_ {
                        return current * *value;
                    } else {
                        set.insert(right.clone());
                        let search_res = search(current * *value, right, final_, map, set);

                        set.remove(right);
                        if search_res != -1.0 {
                            return search_res;
                        };
                    }
                }
                -1.0
            } else {
                -1.0
            }
        }

        let mut set = HashSet::new();
        for q in queries {
            if q[0] == q[1] && map.contains_key(&q[0]) {
                res.push(1.0);
                continue;
            }
            set.insert(q[0].clone());

            res.push(search(1.0, &q[0], &q[1], &map, &mut set));
            set.remove(&q[0]);
        }

        res
    }

    //400
    pub fn find_nth_digit(n: i32) -> i32 {
        let n = n as usize;
        let mut step = 1;
        let mut digit = 0;
        let mut start = 0;
        let mut len = 0;
        for i in 1 as usize.. {
            let next = digit + i * 9 * step;
            if next > n {
                start = step;
                len = i;
                break;
            }
            digit = next;
            step *= 10;
        }
        for i in start.. {
            digit += len;
            if digit >= n {
                let offset = digit - n;
                let i = i.to_string();

                return i.as_bytes()[i.len() - offset] as i32 - 48;
            }
        }
        0
    }
}
