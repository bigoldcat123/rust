use std::collections::{HashSet, VecDeque};



pub fn shortest_path(grid: Vec<Vec<i32>>, k: i32) -> i32 {
    use std::collections::{VecDeque,HashSet};

    let mut qs = vec![(VecDeque::new());k as usize + 1];
    qs[k as usize].push_back((0_i32,0_i32));
    let mut vis = HashSet::new();
    vis.insert((0_i32,0_i32));
            let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut ans = 0;
    while qs.iter().any(|x| !x.is_empty()) {
        println!("{:?}",qs);
        for q_i in 0..qs.len() {
            let pq = qs[q_i].split_off(0);
            for (i,j) in pq {
                if i as usize== grid.len() - 1&& j as usize == grid[0].len() - 1 {
                    return ans
                }
                for (di,dj) in d.iter() {
                    let i = di + i;
                    let j = dj + j;
                    if i >= 0 && (i as usize) < grid.len() && j >= 0 && (j as usize) < grid[0].len() && vis.insert((i,j)) {
                        if grid[i as usize][j as usize] == 0 {
                            qs[q_i].push_back((i,j));
                        }else if q_i > 0 {
                            qs[q_i - 1].push_back((i,j));
                        }
                    }
                }
            }
        }
        ans += 1;
    }
    -1
}


pub fn highest_ranked_k_items(grid: Vec<Vec<i32>>, pricing: Vec<i32>, start: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
    use std::collections::{HashSet, VecDeque};
    let mut ans = vec![];
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    q.push_back((start[0],start[1]));
    vis.insert((start[0],start[1]));
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut des = 0;
    while !q.is_empty() {
        q.make_contiguous().sort_by(|&(i1,j1),&(i2,j2)| {
            grid[i1 as usize][j1 as usize].cmp(&grid[i2 as usize][j2 as usize])
                .then(i1.cmp(&i2))
                .then(j1.cmp(&j2)).reverse()
        });
        for (i,j) in q.split_off(0) {
            for &(di,dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0 && (i as usize) < grid.len() && j >= 0 && (j as usize) < grid[0].len() && vis.insert((i,j)) && grid[i as usize][j as usize] != 0 {
                    if ans.len() == k as usize {
                        return ans
                    }
                    if grid[i as usize][j as usize] != 1 && grid[i as usize][j as usize] >= pricing[0] && grid[i as usize][j as usize]  <= pricing[1] {
                        ans.push(vec!(i,j));
                    }
                }
            }
        }
        des += 1;
    }
    ans
}

pub fn shortest_bridge(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..grid.len() {
        for j in 0..grid[0].len() {
            if grid[i][j] == 1 {
                dfs(i as i32, j as i32, &grid, &mut q,&mut vis);
            }
        }
        if !q.is_empty() {
            break;
        }
    }
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    let mut ans = 0;
    while !q.is_empty() {
        for (i,j) in q.split_off(0) {
            for &(di,dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0 && (i as usize) < grid.len() && j >= 0 && (j as usize) < grid[0].len() && vis.insert((i,j)) {
                    if grid[i as usize][j as usize] == 1  {
                        return ans
                    }
                    q.push_back((i,j));
                }
            }
        }
        ans += 1;
    }
    ans
}

fn dfs(
    i: i32,
    j: i32,
    grid: &Vec<Vec<i32>>,
    q: &mut VecDeque<(i32, i32)>,
    vis: &mut HashSet<(i32, i32)>,
) {
    if i >= 0 && (i as usize) < grid.len() && j >= 0 && (j as usize) < grid[0].len() && vis.insert((i,j)) && grid[i as usize][j as usize] == 1 {
        q.push_back((i,j));
        let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
        for (di,dj) in d {
            dfs(i + di, j + dj, grid, q, vis);
        }
    }
}

pub fn highest_peak(mut is_water: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    use std::collections::{HashSet, VecDeque};

    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..is_water.len() {
        for j in 0..is_water[0].len() {
            q.push_back((i as i32, j as i32));
            vis.insert((i as i32, j as i32));
        }
    }
    let mut h = 0;
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];

    while !q.is_empty() {
        for (i, j) in q.split_off(0) {
            is_water[i as usize][j as usize] = h;
            for &(di, dj) in d.iter() {
                let i = i + di;
                let j = j + dj;
                if i >= 0
                    && (i as usize) < is_water.len()
                    && j >= 0
                    && (j as usize) < is_water[0].len()
                    && vis.insert((i, j))
                {
                    q.push_back((i, j));
                }
            }
        }
        h += 1;
    }

    is_water
}
pub fn oranges_rotting(mut mat: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    let mut ans = 0;
    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            if mat[i][j] == 2 {
                q.push_back((i as i32, j as i32));
                vis.insert((i as i32, j as i32));
            }
        }
    }

    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    while !q.is_empty() {
        let pq = q.split_off(0);
        for (i, j) in pq {
            for &(di, dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0
                    && (i as usize) < mat.len()
                    && j >= 0
                    && (j as usize) < mat[0].len()
                    && mat[i as usize][j as usize] == 1
                    && vis.insert((i, j))
                {
                    mat[i as usize][j as usize] = 2;
                    q.push_back((i, j));
                }
            }
        }
        ans += 1;
    }
    if mat.iter().flatten().all(|&x| x == 0 || x == 2) {
        ans
    } else {
        -1
    }
}

pub fn update_matrix(mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    use std::collections::{HashSet, VecDeque};
    let mut ans = vec![vec![0; mat[0].len()]; mat.len()];
    let mut q = VecDeque::new();
    let mut vis = HashSet::new();
    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            if mat[i][j] == 0 {
                q.push_back((i as i32, j as i32));
                vis.insert((i as i32, j as i32));
            }
        }
    }
    let d = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    while !q.is_empty() {
        let pq = q.split_off(0);
        for (i, j) in pq {
            let n = ans[i as usize][j as usize];
            for &(di, dj) in d.iter() {
                let i = di + i;
                let j = dj + j;
                if i >= 0
                    && (i as usize) < mat.len()
                    && j >= 0
                    && (j as usize) < mat[0].len()
                    && vis.insert((i, j))
                {
                    ans[i as usize][j as usize] = n + 1;
                    q.push_back((i, j));
                }
            }
        }
    }

    ans
}

pub fn count_mentions(number_of_users: i32, mut events: Vec<Vec<String>>) -> Vec<i32> {
    use std::collections::BTreeSet;
    events.sort_by(|a, b| {
        a[1].parse::<i32>()
            .unwrap()
            .cmp(&b[1].parse::<i32>().unwrap())
            .then(a[0].cmp(&b[0]).reverse())
    });

    let mut is_online = vec![true; number_of_users as usize];
    let mut set = BTreeSet::new(); // (time_stamp,id)
    let mut mentions = vec![0; number_of_users as usize];

    for e in events {
        let time_stamp = e[1].parse::<i32>().unwrap();
        match e[0].as_str() {
            "OFFLINE" => {
                let off_line_id = e[2].parse::<usize>().unwrap();
                is_online[off_line_id] = false;
                set.insert((time_stamp + 60, off_line_id));
            }
            "MESSAGE" => match e[2].as_str() {
                "ALL" => {
                    mentions.iter_mut().for_each(|x| *x += 1);
                }
                "HERE" => {
                    while let Some((time_to_online, p_id)) = set.first() {
                        if time_stamp >= *time_to_online {
                            is_online[*p_id] = true;
                            set.pop_first();
                        }
                    }
                    mentions
                        .iter_mut()
                        .zip(is_online.iter())
                        .for_each(|(p, &is_online)| {
                            if is_online {
                                *p += 1;
                            }
                        });
                }
                other => {
                    for id in other.split_whitespace() {
                        let id = id[2..].parse::<usize>().unwrap();
                        mentions[id] += 1;
                    }
                }
            },
            _ => {
                unreachable!()
            }
        }
    }
    mentions
}
pub fn max_distance(grid: Vec<Vec<i32>>) -> i32 {
    let mut ones = vec![];
    let mut zeroes = vec![];
    for i in 0..grid.len() {
        for j in 0..grid.len() {
            if grid[i][j] == 0 {
                zeroes.push((i as i32, j as i32));
            } else {
                ones.push((i as i32, j as i32));
            }
        }
    }
    if ones.is_empty() || zeroes.is_empty() {
        return -1;
    }
    let mut ans = 0;
    for (i, j) in zeroes {
        let mut min = i32::MAX;
        for &(ii, jj) in ones.iter() {
            min = min.min((i - ii).abs() + (j - jj).abs());
        }
        ans = ans.max(min);
    }
    ans
}
pub fn shortest_path_binary_matrix(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut visited = HashSet::new();
    let mut q = VecDeque::new();
    if grid[0][0] == 0 {
        visited.insert((0_i32, 0_i32));
        q.push_back((0_i32, 0_i32));
    }
    let mut path = 0;
    let x: Vec<(i32, i32)> = vec![
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, -1),
        (-1, 1),
    ];

    while !q.is_empty() {
        let mut pq = q.split_off(0);
        for (i, j) in pq {
            if i as usize == grid.len() - 1 && j as usize == grid[0].len() - 1 {
                return path;
            }
            for &(ii, jj) in x.iter() {
                let (i, j) = (i + ii, j + jj);
                if !(i < 0 || j < 0 || i as usize >= grid.len() || j as usize >= grid[0].len()) {
                    if grid[i as usize][j as usize] == 0 && visited.insert((i, j)) {
                        q.push_back((i, j));
                    }
                }
            }
        }
        path += 1;
    }
    -1
}

pub fn nearest_exit(maze: Vec<Vec<char>>, entrance: Vec<i32>) -> i32 {
    use std::collections::{HashSet, VecDeque};
    let mut visited = HashSet::new();
    let mut q = VecDeque::new();
    let x = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
    let i = entrance[0];
    let j = entrance[1];
    for (ii, jj) in x {
        let (i, j) = (i + ii, j + jj);
        if !(i < 0 || j < 0 || i as usize >= maze.len() || j as usize >= maze[0].len()) {
            if maze[i as usize][j as usize] != '+' {
                if visited.insert((i, j)) {
                    q.push_back((i, j));
                }
            }
        }
    }
    let mut step = -1;
    while !q.is_empty() {
        let mut pq = q.split_off(0);
        for (i, j) in pq {
            if i < 0 || j < 0 || i as usize >= maze.len() || j as usize >= maze[0].len() {
                return step;
            }
            if maze[i as usize][j as usize] != '+' {
                let x = vec![(0, 1), (0, -1), (1, 0), (-1, 0)];
                for (ii, jj) in x {
                    let next = (i + ii, j + jj);
                    if visited.insert(next) {
                        q.push_back(next);
                    }
                }
            }
        }
        step += 1;
    }
    step
}
