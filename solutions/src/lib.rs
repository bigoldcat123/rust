use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
};

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
                println!("{}-{:?}",ele.0,e.val);
            }else {
                 println!("{:?} is none",ele.0);
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
                    resume(&mut next, nodes, n + 1, len,real_len);
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
}
