use std::{i32, i64, iter::Map, num};

pub fn maximums_spliced_array(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let mut pre_sum1 = vec![0; nums1.len() + 1];
    let mut pre_sum2 = vec![0; nums2.len() + 1];
    for i in 0..nums1.len() {
        pre_sum1[i + 1] = pre_sum1[i] + nums1[i];
        pre_sum2[i + 1] = pre_sum2[i] + nums2[i];
    }
    let mut sum1 = nums1.iter().sum::<i32>();
    let mut sum2 = nums2.iter().sum::<i32>();

    let mut diff = nums1
        .iter()
        .zip(nums2.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let mut pre_sum_diff = vec![0; diff.len() + 1];
    for i in 0..diff.len() {
        pre_sum_diff[i + 1] = pre_sum_diff[i] + diff[i];
    }
    let mut dp = vec![0; diff.len()];
    dp[0] = diff[0];
    for i in 1..diff.len() {
        dp[i] = 0.min(dp[i - 1]) + diff[i];
    }
    let mut min_with_end_idx = (i32::MAX, 0);
    for i in 0..diff.len() {
        if dp[i] < min_with_end_idx.0 {
            min_with_end_idx = (dp[i], i);
        }
    }
    let ps = pre_sum_diff[min_with_end_idx.1 + 1];
    let mut res = sum1;
    for i in (0..min_with_end_idx.1).rev() {
        if min_with_end_idx.0 == ps - pre_sum_diff[i] {
            let (start, end) = (i, min_with_end_idx.1);
            let rep = pre_sum2[end] - pre_sum2[start];
            if rep - min_with_end_idx.0 >= 0 {
                res += (rep - min_with_end_idx.0);
            }
        }
    }
    res
}

pub fn max_product(nums: Vec<i32>) -> i32 {
    let mut dp = vec![0; nums.len()];
    let mut dp2 = vec![0; nums.len()];
    dp[0] = nums[0];
    for i in 1..nums.len() {
        dp[i] = (dp[i - 1] * nums[i]).max(nums[i]).max(dp2[i - 1] * nums[i]);
        dp2[i] = (dp2[i - 1] * nums[i]).min(nums[i]).min(dp[i - 1] * nums[i]);
    }
    dp.into_iter().max().unwrap()
}

pub fn maximum_sum(arr: Vec<i32>) -> i32 {
    let mut dp = vec![vec![0; arr.len()]; 2];
    dp[0][0] = arr[0];
    for i in 1..arr.len() {
        dp[0][i] = 0.max(dp[0][i - 1]) + arr[i];
        dp[1][i] = dp[0][i - 1].max(0.max(dp[1][i - 1]) + arr[i])
    }
    dp[0]
        .iter()
        .max()
        .copied()
        .unwrap()
        .max(dp[1][1..].iter().max().copied().unwrap_or(i32::MIN))
}

pub fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let mut dp = grid.clone();

    for i in 1..dp.len() {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }
    for j in 1..dp[0].len() {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]) + grid[i][j];
        }
    }

    *dp.last().unwrap().last().unwrap()
}

pub fn unique_paths(m: i32, n: i32) -> i32 {
    let mut dp = vec![vec![1; n as usize]; m as usize];

    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            dp[i][j] = dp[i - 1][j] + (dp[i][j - 1]);
        }
    }

    *dp.last().unwrap().last().unwrap()
}
pub fn unique_paths_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![0; obstacle_grid[0].len()]; obstacle_grid.len()];
    for i in 0..dp.len() {
        if obstacle_grid[i][0] == 1 {
            break;
        }
        dp[i][0] = 1;
    }
    for j in 0..dp[0].len() {
        if obstacle_grid[0][j] == 1 {
            break;
        }
        dp[0][j] = 1;
    }

    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            if obstacle_grid[i][j] == 1 {
                continue;
            }
            dp[i][j] = dp[i - 1][j] + (dp[i][j - 1]);
        }
    }

    *dp.last().unwrap().last().unwrap()
}

pub fn minimum_total(triangle: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![0; triangle.last().unwrap().len()]; triangle.len()];

    for i in 0..triangle.len() {
        dp[i][0] = triangle[i][0];
    }
    for i in 1..triangle.len() {
        for j in 1..i + 1 {
            dp[i][j] = dp[i - 1][j - 1].min(dp[i - 1][j]) + triangle[i][j];
        }
    }
    *dp.last().unwrap().iter().min().unwrap()
}

pub fn count_paths_with_xor_value(grid: Vec<Vec<i32>>, k: i32) -> i32 {
    let mut dp = vec![vec![[0; 16]; grid[0].len()]; grid.len()];
    dp[0][0][grid[0][0] as usize] = 1;
    for i in 1..dp.len() {
        let mut next = [0; 16];
        for (i, x) in dp[i - 1][0].iter().enumerate() {
            if *x != 0 {
                next[i ^ (grid[i][0] as usize)] += *x;
            }
        }
        dp[i][0] = next;
    }

    for j in 1..dp[0].len() {
        let mut next = [0; 16];
        for (i, x) in dp[0][j - 1].iter().enumerate() {
            if *x != 0 {
                next[i ^ (grid[0][j] as usize)] += *x;
            }
        }
        dp[0][j] = next;
    }
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            let mut next = [0; 16];
            for (i, x) in dp[i - 1][j].iter().enumerate() {
                if *x != 0 {
                    next[i ^ (grid[i][j] as usize)] += *x;
                }
            }
            for (i, x) in dp[i][j - 1].iter().enumerate() {
                if *x != 0 {
                    next[i ^ (grid[i][j] as usize)] += *x;
                }
            }
            dp[i][j] = next;
        }
    }
    dp.last().unwrap().last().unwrap()[k as usize]
}

pub fn min_falling_path_sum(matrix: Vec<Vec<i32>>) -> i32 {
    let mut dp = matrix.clone();
    for i in 1..dp.len() {
        for j in 0..dp[0].len() {
            let mut min = dp[i - 1][j];
            if j > 0 {
                min = min.min(dp[i - 1][j - 1]);
            }
            if j < dp[0].len() - 1 {
                min = min.min(dp[i - 1][j + 1]);
            }
            dp[i][j] += min;
        }
    }
    dp.last().unwrap().iter().min().copied().unwrap()
}

pub fn min_cost(m: i32, n: i32, wait_cost: Vec<Vec<i32>>) -> i64 {
    let mut dp = vec![vec![1; n as usize]; m as usize];

    for i in 1..dp.len() {
        dp[i][0] = (dp[i - 1][0] + (i as i64 + 1) + wait_cost[i][0] as i64);
    }
    for j in 1..dp[0].len() {
        dp[0][j] = (dp[0][j - 1] + (j as i64 + 1) + wait_cost[0][j] as i64);
    }
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            dp[i][j] = (dp[i - 1][j].min(dp[i][j - 1])
                + (i as i64 + 1) * (j as i64 + 1)
                + wait_cost[i][j] as i64);
        }
    }
    dp.last().unwrap().last().copied().unwrap()
        - wait_cost.last().unwrap().last().copied().unwrap() as i64
}

pub fn min_path_cost(grid: Vec<Vec<i32>>, move_cost: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![0; grid[0].len()]; grid.len() - 1];

    for i in 0..dp[0].len() {
        let mut min = i32::MAX;
        for j in 0..dp[0].len() {
            min = min.min(move_cost[grid[0][j] as usize][i] + grid[0][j]);
        }
        dp[0][i] = min;
    }
    for i in 1..dp.len() {
        for j in 0..dp[0].len() {
            let mut min = i32::MAX;
            for k in 0..dp[0].len() {
                min = min.min(dp[i - 1][k] + move_cost[grid[i][k] as usize][j] + grid[i][k])
            }
            dp[i][j] = min;
        }
    }
    let mut res = i32::MAX;
    for (i, v) in dp.last().unwrap().iter().enumerate() {
        res = res.min(v + grid.last().unwrap()[i])
    }
    res
}

pub fn min_falling_path_sum2(grid: Vec<Vec<i32>>) -> i32 {
    let mut dp = grid.clone();
    for i in 1..dp.len() {
        for j in 0..dp[0].len() {
            dp[i][j] = i32::MAX;
            for k in 0..dp[0].len() {
                if k != j {
                    dp[i][j] = (dp[i - 1][k] + grid[i][j]).min(dp[i][j]);
                }
            }
        }
    }
    dp.last().unwrap().iter().min().copied().unwrap()
}

pub fn min_cost2(grid: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashSet;
    let mut dp = vec![vec![HashSet::new(); grid[0].len()]; grid.len()];

    dp[0][0].insert(grid[0][0]);
    for i in 1..dp.len() {
        let mut set = HashSet::new();
        for item in dp[i - 1][0].iter() {
            set.insert(grid[i][0] ^ item);
        }
        dp[i][0] = set;
    }
    for i in 1..dp[0].len() {
        let mut set = HashSet::new();
        for item in dp[0][i - 1].iter() {
            set.insert(grid[0][i] ^ item);
        }
        dp[0][i] = set;
    }
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            let mut set = HashSet::new();
            for item in dp[i - 1][j].iter() {
                set.insert(grid[i][j] ^ item);
            }
            for item in dp[i][j - 1].iter() {
                set.insert(grid[i][j] ^ item);
            }
            dp[i][j] = set;
        }
    }

    dp.last().unwrap().iter().flatten().min().copied().unwrap()
}

pub fn maximum_amount(coins: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![(None, None, None); coins[0].len()]; coins.len()];
    if coins[0][0] < 0 {
        dp[0][0].1 = Some(0)
    }
    dp[0][0].0 = Some(coins[0][0]);

    for i in 1..coins.len() {
        if coins[i][0] < 0 {
            dp[i][0].1 = dp[i - 1][0].0.max(dp[i - 1][0].1.map(|x| x + coins[i][0]));
            dp[i][0].2 = dp[i - 1][0].1.max(dp[i - 1][0].2.map(|x| x + coins[i][0]));
        } else {
            dp[i][0].1 = dp[i - 1][0].1.map(|x| x + coins[i][0]);

            dp[i][0].2 = dp[i - 1][0].2.map(|x| x + coins[i][0]);
        }
        dp[i][0].0 = dp[i - 1][0].0.map(|x| x + coins[i][0]);
    }
    for j in 1..coins[0].len() {
        if coins[0][j] < 0 {
            dp[0][j].1 = dp[0][j - 1].0.max(dp[0][j - 1].1).map(|x| x + coins[0][j]);
            dp[0][j].2 = dp[0][j - 1].1.max(dp[0][j - 1].2.map(|x| x + coins[0][j]));
        } else {
            dp[0][j].1 = dp[0][j - 1].1.map(|x| x + coins[0][j]);

            dp[0][j].2 = dp[0][j - 1].2.map(|x| x + coins[0][j]);
        }
        dp[0][j].0 = dp[0][j - 1].0.map(|x| x + coins[0][j]);
    }
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            if coins[i][j] < 0 {
                dp[i][j].1 = dp[i - 1][j]
                    .0
                    .max(dp[i - 1][j].1.map(|x| x + coins[i][j]))
                    .max(dp[i][j - 1].0)
                    .max(dp[i][j - 1].1.map(|x| x + coins[i][j]));
                dp[i][j].2 = dp[i - 1][j]
                    .1
                    .max(dp[i - 1][j].2.map(|x| x + coins[i][j]))
                    .max(dp[i][j - 1].1)
                    .max(dp[i][j - 1].2.map(|x| x + coins[i][j]));
            } else {
                dp[i][j].1 = dp[i - 1][j]
                    .1
                    .map(|x| x + coins[i][j])
                    .max(dp[i][j - 1].1.map(|x| x + coins[i][j]));
                dp[i][j].2 = dp[i - 1][j]
                    .2
                    .map(|x| x + coins[i][j])
                    .max(dp[i][j - 1].2.map(|x| x + coins[i][j]));
            }
            dp[i][j].0 = dp[i - 1][j]
                .0
                .map(|x| x + coins[i][j])
                .max(dp[i][j - 1].0.map(|x| x + coins[i][j]));
        }
    }
    // for i in dp.iter() {
    //     println!("{i:?}");
    // }
    let last = dp.last().unwrap().last().unwrap();
    last.0.max(last.1).max(last.2).unwrap_or(0)
}

pub fn max_path_score(grid: Vec<Vec<i32>>, k: i32) -> i32 {
    let mut dp = vec![vec![(0, k); grid[0].len()]; grid.len()];
    for i in 1..dp[0].len() {
        dp[0][i].0 = dp[0][i - 1].0 + grid[0][i];
        dp[0][i].1 = dp[0][i - 1].1 - grid[0][i].min(1);
    }

    for i in 1..dp.len() {
        dp[i][0].0 = dp[i - 1][0].0 + grid[i][0];
        dp[i][0].1 = dp[i - 1][0].1 - grid[i][0].min(1);
    }
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            let current = grid[i][j];
            let cost = grid[i][j].min(1);
            let (up_score, up_k) = dp[i - 1][j];
            let (left_score, left_k) = dp[i][j - 1];
            if up_k - cost >= 0 && left_k - cost >= 0 {
                if up_score > left_score {
                    dp[i][j].0 = up_score + current;
                    dp[i][j].1 = up_k - cost;
                } else if up_score == left_score {
                    dp[i][j].0 = up_score + current;
                    dp[i][j].1 = up_k.max(left_k) - cost;
                } else {
                    dp[i][j].0 = left_score + current;
                    dp[i][j].1 = left_k - cost;
                }
            } else if up_k - cost < 0 && left_k - cost >= 0 {
                dp[i][j].0 = left_score + current;
                dp[i][j].1 = left_k - cost;
            } else if up_k - cost >= 0 && left_k - cost < 0 {
                dp[i][j].0 = up_score + current;
                dp[i][j].1 = up_k.max(left_k) - cost;
            } else {
                dp[i][j] = (0, -1);
            }
        }
    }
    let a = dp.last().unwrap().last().unwrap().0;
    let b = dp.last().unwrap().last().unwrap().1;
    if b < 0 { -1 } else { a }
}

pub fn max_moves(grid: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![0; grid[0].len()]; grid.len()];
    for j in 1..grid[0].len() {
        let mut ok = false;
        for i in 0..grid.len() {
            if i > 0 {
                if grid[i][j] > grid[i - 1][j - 1] && dp[i - 1][j - 1] == j as i32 - 1 {
                    dp[i][j] = j as i32;
                    ok = true
                }
            }
            if i < grid.len() - 1 {
                if grid[i][j] > grid[i + 1][j - 1] && dp[i + 1][j - 1] == j as i32 - 1 {
                    dp[i][j] = j as i32;
                    ok = true
                }
            }
            if grid[i][j] > grid[i][j - 1] && dp[i][j - 1] == j as i32 - 1 {
                dp[i][j] = j as i32;
                ok = true
            }
        }
        if !ok {
            return j as i32 - 1;
        }
    }
    grid[0].len() as i32 - 1
}

pub fn min_side_jumps(obstacles: Vec<i32>) -> i32 {
    let mut dp = vec![vec![0; 3]; obstacles.len()];
    dp[0][1] = 0;
    dp[0][0] = 1;
    dp[0][2] = 1;
    for i in 1..dp.len() {
        if obstacles[i] == 0 {
            dp[i][0] = dp[i - 1][0].min(dp[i - 1][1].min(dp[i - 1][2]) + 1);
            dp[i][1] = dp[i - 1][1].min(dp[i - 1][0].min(dp[i - 1][2]) + 1);
            dp[i][2] = dp[i - 1][2].min(dp[i - 1][0].min(dp[i - 1][1]) + 1);
        } else if obstacles[i] == 1 {
            dp[i][0] = i32::MAX;
            dp[i][1] = dp[i - 1][1].min(dp[i - 1][0].min(dp[i - 1][2]) + 1);
            dp[i][2] = dp[i - 1][2].min(dp[i - 1][0].min(dp[i - 1][1]) + 1);
        } else if obstacles[i] == 2 {
            dp[i][0] = dp[i - 1][0].min(dp[i - 1][1].min(dp[i - 1][2]) + 1);
            dp[i][1] = i32::MAX;
            dp[i][2] = dp[i - 1][2].min(dp[i - 1][0].min(dp[i - 1][1]) + 1);
        } else if obstacles[i] == 3 {
            dp[i][0] = dp[i - 1][0].min(dp[i - 1][1].min(dp[i - 1][2]) + 1);
            dp[i][1] = dp[i - 1][1].min(dp[i - 1][0].min(dp[i - 1][2]) + 1);
            dp[i][2] = i32::MAX;
        }
    }
    let d = dp.last().unwrap();
    d[0].min(d[1]).min(d[2])
}

pub fn max_product_path(grid: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![];
    dp.push(vec![vec![-1_i128; grid[0].len()]; grid.len()]);
    dp.push(vec![vec![1_i128; grid[0].len()]; grid.len()]);
    // dp[k][i][j]
    if grid[0][0] >= 0 {
        dp[0][0][0] = grid[0][0] as i128;
    } else {
        dp[1][0][0] = grid[0][0] as i128;
    }
    for i in 1..grid.len() {
        if grid[i][0] >= 0 {
            dp[0][i][0] = dp[0][i - 1][0] * grid[i][0] as i128;
            dp[1][i][0] = dp[1][i - 1][0] * grid[i][0] as i128;
        } else {
            dp[0][i][0] = dp[1][i - 1][0] * grid[i][0] as i128;
            dp[1][i][0] = dp[0][i - 1][0] * grid[i][0] as i128;
        }
    }
    for i in 1..grid[0].len() {
        if grid[0][i] >= 0 {
            dp[0][0][i] = dp[0][0][i - 1] * grid[0][i] as i128;
            dp[1][0][i] = dp[1][0][i - 1] * grid[0][i] as i128;
        } else {
            dp[0][0][i] = dp[1][0][i - 1] * grid[0][i] as i128;
            dp[1][0][i] = dp[0][0][i - 1] * grid[0][i] as i128;
        }
    }
    for i in 1..grid.len() {
        for j in 1..grid[0].len() {
            if grid[i][j] >= 0 {
                dp[0][i][j] = dp[0][i - 1][j].max(dp[0][i][j - 1]) * grid[i][j] as i128;
                dp[1][i][j] = dp[1][i - 1][j].min(dp[1][i][j - 1]) * grid[i][j] as i128;
            } else {
                dp[0][i][j] = dp[1][i - 1][j].min(dp[1][i][j - 1]).min(1) * grid[i][j] as i128;
                dp[1][i][j] = dp[0][i - 1][j].max(dp[0][i][j - 1]).max(-1) * grid[i][j] as i128;
            }
        }
    }
    // for x in dp.iter() {
    //     println!("{x:?}");
    // }
    let res = dp[0].last().unwrap().last().unwrap();
    if *res >= 0 {
        (*res % 1_000_000_007) as i32
    } else {
        -1
    }
}

pub fn number_of_paths(grid: Vec<Vec<i32>>, k: i32) -> i32 {
    let mut dp = vec![vec![vec![0; k as usize]; grid[0].len()]; grid.len()];

    dp.last().unwrap().last().unwrap()[0]
}
