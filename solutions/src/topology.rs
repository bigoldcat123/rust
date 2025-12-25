use std::{i32, vec};

fn cal_order(k: i32, conditions: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::{HashMap, VecDeque};
    let mut dep_map: HashMap<i32, Vec<i32>> = HashMap::new();

    let mut in_degree = vec![0; k as usize + 1];
    for row_cnd in conditions.iter() {
        in_degree[row_cnd[0] as usize] += 1;
        dep_map.entry(row_cnd[1]).or_default().push(row_cnd[0]);
    }
    let mut q = VecDeque::new();
    for i in 1..in_degree.len() {
        if in_degree[i] == 0 {
            q.push_back(i as i32);
            in_degree[i] = i32::MAX;
        }
    }
    let mut order = vec![];
    while let Some(n) = q.pop_front() {
        order.push(n);
        if let Some(next) = dep_map.get(&n) {
            for &n in next {
                in_degree[n as usize] -= 1;
            }
            for i in 1..in_degree.len() {
                if in_degree[i] == 0 {
                    q.push_back(i as i32);
                    in_degree[i] = i32::MAX;
                }
            }
        }
    }
    println!("{:?}",order);

    order
}
pub fn build_matrix(
    k: i32,
    row_conditions: Vec<Vec<i32>>,
    col_conditions: Vec<Vec<i32>>,
) -> Vec<Vec<i32>> {
    let row_order = cal_order(k, row_conditions);
    let col_order = cal_order(k, col_conditions);
    if row_order.is_empty() || col_order.is_empty() {
        return vec![];
    }
    let mut idx = vec![0;k as usize + 1];
    let mut ans = vec![
        vec![0;k as usize];k as usize
    ];
    for (i,n) in row_order.into_iter().enumerate() {
        ans[k as usize - i - 1][k as usize - 1] = n;
        idx[n as usize + 1] = k as usize - i - 1;
    }
    for (j ,n) in col_order.into_iter().enumerate() {
        let i = idx[n as usize - 1];
        ans[i][k as usize - 1] = 0;
        ans[i][k as usize - 1 - i] = n;
    }
    ans
}

pub fn find_all_recipes(
    recipes: Vec<String>,
    ingredients: Vec<Vec<String>>,
    supplies: Vec<String>,
) -> Vec<String> {
    use std::collections::{HashMap, VecDeque};
    let mut ingredient_map: HashMap<&String, Vec<&String>> = HashMap::new();
    for (i, ingredient) in ingredients.iter().enumerate() {
        for ig in ingredient {
            ingredient_map.entry(ig).or_default().push(&recipes[i]);
        }
    }
    let mut in_deg = ingredients
        .iter()
        .enumerate()
        .map(|x| (&recipes[x.0], x.1.len()))
        .collect::<HashMap<&String, usize>>();
    for s in supplies {
        if let Some(e) = ingredient_map.get(&&s) {
            for food in e {
                in_deg.entry(food).and_modify(|v| *v -= 1);
            }
        }
    }
    let mut q = VecDeque::new();
    for (food, ideg) in in_deg.iter() {
        if *ideg == 0 {
            q.push_back(*food);
        }
    }
    let mut ans = vec![];
    while !q.is_empty() {
        for f in q.split_off(0) {
            ans.push(f.clone());
            if let Some(e) = ingredient_map.get(f) {
                for food in e {
                    in_deg.entry(food).and_modify(|v| *v -= 1);
                }
                for (food, ideg) in in_deg.iter() {
                    if *ideg == 0 {
                        q.push_back(*food);
                    }
                }
            }
        }
    }
    ans
}

pub fn find_order(num_courses: i32, prerequisites: Vec<Vec<i32>>) -> Vec<i32> {
    use std::collections::VecDeque;
    let mut in_degree = vec![0; num_courses as usize];
    let mut next = vec![vec![]; num_courses as usize];
    for pre in prerequisites {
        in_degree[pre[0] as usize] += 1;
        next[pre[1] as usize].push(pre[0]);
    }
    let mut q = VecDeque::new();

    let mut ans = vec![];
    for (i, &in_d) in in_degree.iter().enumerate() {
        if in_d == 0 {
            q.push_back(i);
        }
    }

    while let Some(n) = q.pop_front() {
        ans.push(n as i32);
        for &next in next[n].iter() {
            in_degree[next as usize] -= 1;
            if in_degree[next as usize] == 0 {
                q.push_back(next as usize);
            }
        }
    }

    if ans.len() != num_courses as usize {
        vec![]
    } else {
        ans
    }
}
