use std::{
    i32, i64,
    iter::Map,
    num::{self, FpCategory},
    os::unix::raw::uid_t,
    result::Iter,
};

use num_traits::sign;
use rand::rand_core::le;

use crate::{dijkstra, dijoint_set::friend_requests};

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
    dp[0][0][(grid[0][0] % k) as usize] = 1;
    const MOD: i32 = 1_000_000_007;
    for i in 1..grid.len() {
        let x = (grid[i][0] % k) as usize;
        for kk in 0..k as usize {
            dp[i][0][(kk + x) % k as usize] += dp[i - 1][0][kk];
        }
    }
    for j in 1..grid[0].len() {
        let x = (grid[0][j] % k) as usize;
        for kk in 0..k as usize {
            dp[0][j][(kk + x) % k as usize] += dp[0][j - 1][kk];
        }
    }
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            let x = (grid[i][j] % k) as usize;
            for kk in 0..k as usize {
                dp[i][j][(kk + x) % k as usize] += dp[i - 1][j][kk];
                dp[i][j][(kk + x) % k as usize] %= MOD;
                dp[i][j][(kk + x) % k as usize] += dp[i][j - 1][kk];
                dp[i][j][(kk + x) % k as usize] %= MOD;
            }
        }
    }
    for j in 1..grid[0].len() {}
    dp.last().unwrap().last().unwrap()[0]
}

pub fn calculate_minimum_hp(dungeon: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashMap;
    let mut dp = vec![vec![HashMap::new(); dungeon[0].len()]; dungeon.len()];
    let first_ele = dungeon[0][0];
    let back = if first_ele >= 0 { first_ele } else { 0 };
    let init = (1 - first_ele).max(1);
    dp[0][0].insert(back, init);
    for i in 1..dp.len() {
        if dungeon[i][0] >= 0 {
            for (back, life) in dp[i - 1][0].clone() {
                dp[i][0].insert(back + dungeon[i][0], life);
            }
        } else {
            let cost = -dungeon[i][0] + 1;
            for (back, life) in dp[i - 1][0].clone() {
                if cost <= back {
                    dp[i][0].insert(back - cost, life);
                } else {
                    let diff = cost - back;
                    if let Some(current_life) = dp[i][0].get_mut(&0) {
                        if *current_life > life + diff {
                            *current_life = life + diff;
                        }
                    } else {
                        dp[i][0].insert(0, life + diff);
                    }
                }
            }
        }
    }
    for j in 1..dp[0].len() {
        if dungeon[0][j] >= 0 {
            for (back, life) in dp[0][j - 1].clone() {
                dp[0][j].insert(back + dungeon[0][j], life);
            }
        } else {
            let cost = -dungeon[0][j] + 1;
            for (back, life) in dp[0][j - 1].clone() {
                if cost <= back {
                    dp[0][j].insert(back - cost, life);
                } else {
                    let diff = cost - back;
                    if let Some(current_life) = dp[0][j].get_mut(&0) {
                        if *current_life > life + diff {
                            *current_life = life + diff;
                        }
                    } else {
                        dp[0][j].insert(0, life + diff);
                    }
                }
            }
        }
    }

    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            if dungeon[i][j] >= 0 {
                for (back, life) in dp[i - 1][j].clone() {
                    dp[i][j].insert(back + dungeon[i][j], life);
                }
            } else {
                let cost = -dungeon[i][j] + 1;
                for (back, life) in dp[i - 1][j].clone() {
                    if cost <= back {
                        dp[i][j].insert(back - cost, life);
                    } else {
                        let diff = cost - back;
                        if let Some(current_life) = dp[i][j].get_mut(&0) {
                            if *current_life > life + diff {
                                *current_life = life + diff;
                            }
                        } else {
                            dp[i][j].insert(0, life + diff);
                        }
                    }
                }
            }

            if dungeon[i][j] >= 0 {
                // let mut p = HashMap::new();
                for (back, life) in dp[i][j - 1].clone() {
                    if let Some(x) = dp[i][j].get_mut(&(back + dungeon[i][j])) {
                        if life < *x {
                            *x = life;
                        }
                    } else {
                        dp[i][j].insert(back + dungeon[i][j], life);
                    }
                }
                // dp[i][j].extend(p);
            } else {
                let cost = -dungeon[i][j] + 1;
                for (back, life) in dp[i][j - 1].clone() {
                    if cost <= back {
                        if let Some(x) = dp[i][j].get_mut(&(back - cost)) {
                            if life < *x {
                                *x = life;
                            }
                        } else {
                            dp[i][j].insert(back - cost, life);
                        }
                    } else {
                        let diff = cost - back;
                        if let Some(current_life) = dp[i][j].get_mut(&0) {
                            if *current_life > life + diff {
                                *current_life = life + diff;
                            }
                        } else {
                            dp[i][j].insert(0, life + diff);
                        }
                    }
                }
            }
        }
    }
    let mut min = i32::MAX;
    for l in dp.last().unwrap().last().unwrap().values() {
        min = min.min(*l);
    }
    min
}

pub fn longest_increasing_path(matrix: Vec<Vec<i32>>) -> i32 {
    let mut visited = vec![vec![false; matrix[0].len()]; matrix.len()];
    let mut dp = vec![vec![0; matrix[0].len()]; matrix.len()];
    let mut max = 0;
    for i in 0..dp.len() {
        for j in 0..dp[0].len() {
            if !visited[i][j] {
                max = max.max(deep_search_longest_increasing_path(
                    i,
                    j,
                    &mut visited,
                    &mut dp,
                    &matrix,
                ));
            }
        }
    }
    max
}

fn deep_search_longest_increasing_path(
    i: usize,
    j: usize,
    visited: &mut [Vec<bool>],
    dp: &mut [Vec<i32>],
    matrix: &Vec<Vec<i32>>,
) -> i32 {
    if visited[i][j] {
        dp[i][j]
    } else {
        visited[i][j] = true;
        let mut max = 1;

        let dir = [
            (i as i32 - 1, j as i32),
            (i as i32 + 1, j as i32),
            (i as i32, j as i32 - 1),
            (i as i32, j as i32 + 1),
        ];
        let current = matrix[i][j];
        for (i, j) in dir {
            if i >= 0 && i < dp.len() as i32 && j >= 0 && j < dp.len() as i32 {
                let (i, j) = (i as usize, j as usize);
                let next = matrix[i][j];
                if next > current {
                    max =
                        max.max(1 + deep_search_longest_increasing_path(i, j, visited, dp, matrix))
                }
            }
        }

        dp[i][j] = max;
        max
    }
}

pub fn count_paths(matrix: Vec<Vec<i32>>) -> i32 {
    let mut visited = vec![vec![false; matrix[0].len()]; matrix.len()];
    let mut dp = vec![vec![0; matrix[0].len()]; matrix.len()];
    let mut res = 0;
    const MOD: i32 = 1_000_000_007;
    for i in 0..dp.len() {
        for j in 0..dp[0].len() {
            if !visited[i][j] {
                res = (res + deep_search_count_paths(i, j, &mut visited, &mut dp, &matrix)) % MOD;
            }
        }
    }
    res
}

fn deep_search_count_paths(
    i: usize,
    j: usize,
    visited: &mut [Vec<bool>],
    dp: &mut [Vec<i32>],
    matrix: &Vec<Vec<i32>>,
) -> i32 {
    if visited[i][j] {
        dp[i][j]
    } else {
        visited[i][j] = true;
        let mut res = 1;

        let dir = [
            (i as i32 - 1, j as i32),
            (i as i32 + 1, j as i32),
            (i as i32, j as i32 - 1),
            (i as i32, j as i32 + 1),
        ];
        let current = matrix[i][j];
        for (i, j) in dir {
            if i >= 0 && i < dp.len() as i32 && j >= 0 && j < dp[0].len() as i32 {
                let (i, j) = (i as usize, j as usize);
                let next = matrix[i][j];
                if next > current {
                    res =
                        (res + deep_search_count_paths(i, j, visited, dp, matrix)) % 1_000_000_007;
                }
            }
        }
        dp[i][j] = res;
        res
    }
}

pub fn has_valid_path(grid: Vec<Vec<char>>) -> bool {
    use std::collections::HashSet;
    let mut dp = vec![vec![HashSet::new(); grid[0].len()]; grid.len()];
    if grid[0][0] == '(' {
        return false;
    } else {
        dp[0][0].insert((1, 0));
    }
    for i in 0..dp.len() {
        for j in 0..dp[0].len() {
            let c = grid[i][j];
            if i > 0 {
                for (l, r) in dp[i - 1][j].clone() {
                    if c == '(' {
                        dp[i][j].insert((l + 1, r));
                    } else {
                        if r + 1 <= l {
                            dp[i][j].insert((l, r + 1));
                        }
                    }
                }
            }
            if j > 0 {
                for (l, r) in dp[i][j - 1].clone() {
                    if c == '(' {
                        dp[i][j].insert((l + 1, r));
                    } else {
                        if r + 1 <= l {
                            dp[i][j].insert((l, r + 1));
                        }
                    }
                }
            }
        }
    }
    dp.last()
        .unwrap()
        .last()
        .unwrap()
        .iter()
        .any(|x| x.0 == x.1)
}

pub fn max_points(points: Vec<Vec<i32>>) -> i64 {
    let mut dp: Vec<Vec<i64>> = points
        .iter()
        .map(|x| x.iter().map(|&x| x as i64).collect())
        .collect();
    let points = dp.clone();
    let mut m = dp[0].iter().max().copied().unwrap();
    for i in 1..dp.len() {
        for j in 0..dp[0].len() {
            let mut max = i64::MIN;
            for k in (0..j).rev() {
                let cost = j as i64 - k as i64;
                if cost >= m {
                    break;
                }
                max = max.max((dp[i - 1][k]) - cost + points[i][j]);
            }
            for k in (j..dp[0].len()) {
                let cost = k as i64 - j as i64;
                if cost >= m {
                    break;
                }
                max = max.max((dp[i - 1][k]) - cost + points[i][j]);
            }
            dp[i][j] = max;
            m = m.max(max);
        }
    }
    *dp.last().unwrap().iter().max().unwrap()
}
pub fn max_collected_fruits(mut fruits: Vec<Vec<i32>>) -> i32 {
    // down right down_right
    // down_left down down_right
    // up_right right up_right
    let mut res = 0;
    for i in 0..fruits.len() {
        res += fruits[i][i];
        fruits[i][i] = 0;
    }
    let n = fruits.len();
    let mut dp = vec![vec![0; 2]; n];
    dp[0][1] = fruits[0][n - 1];
    for i in 1..n {
        dp[i][2 - 1] = fruits[i][n - 1] + dp[i - 1][2 - 1].max(dp[i - 1][2 - 2]);
        dp[i][2 - 2] = fruits[i][n - 2] + dp[i - 1][2 - 1].max(dp[i - 1][2 - 2]);
    }
    res += dp.last().unwrap().last().unwrap();

    let mut dp = vec![vec![0; n]; 2];

    for i in 1..n {
        dp[2 - 1][i] = fruits[n - 1][i] + dp[2 - 1][i - 1].max(dp[2 - 2][i - 1]);
        dp[2 - 2][i] = fruits[n - 2][i] + dp[2 - 1][i - 1].max(dp[2 - 2][i - 1]);
    }
    res += dp.last().unwrap().last().unwrap();
    res
}

pub fn can_partition(nums: Vec<i32>) -> bool {
    let len = nums.iter().sum::<i32>();
    if len % 2 == 0 {
        let len = len as usize / 2;
        let mut dp = vec![vec![0; nums.len()]; len as usize + 1];
        // dp[i][j] i -> pre ith j -> with wight j
        for w in 0..=len {
            if w as i32 >= nums[0] {
                dp[0][w] = nums[0];
            }
        }
        for i in 1..dp.len() {
            for w in 1..=len {
                dp[i][w] = if w as i32 >= nums[i] {
                    dp[i - 1][w].max(dp[i - 1][w - nums[i] as usize] + nums[i])
                } else {
                    dp[i - 1][w]
                }
            }
        }
        *dp.last().unwrap().last().unwrap() == len as i32
    } else {
        false
    }
}

pub fn find_target_sum_ways(nums: Vec<i32>, target: i32) -> i32 {
    let mut dp = vec![vec![0; 2001]; nums.len()];
    //dp[i][j] ith num with target = j
    dp[nums[0] as usize + 1000][0] = 1;
    dp[(-nums[0] + 1000) as usize][0] = 1;
    for i in 1..nums.len() {}

    0
}
fn dfs_find_target_sum_ways(i: usize, nums: &[i32], res: &mut i32, current: i32, target: i32) {
    if i == nums.len() {
        if current == target {
            *res += 1;
        }
    } else {
        dfs_find_target_sum_ways(i + 1, nums, res, current + nums[i], target);
        dfs_find_target_sum_ways(i + 1, nums, res, current - nums[i], target);
    }
}

pub fn length_of_longest_subsequence(nums: Vec<i32>, target: i32) -> i32 {
    let mut dp = vec![vec![0; target as usize + 1]; nums.len()];
    dp[0][nums[0] as usize] = 1;
    for i in 1..nums.len() {
        for t in 1..=target as usize {
            dp[i][t] = if nums[i] <= t as i32 {
                dp[i - 1][t].max(
                    1 + if dp[i - 1][t - nums[i] as usize] == 0 {
                        i32::MIN
                    } else {
                        dp[i - 1][t - nums[i] as usize]
                    },
                )
            } else {
                dp[i - 1][t]
            }
        }
    }
    for d in dp.iter() {
        println!("{d:?}");
    }
    if dp[nums.len() - 1][target as usize] == 0 {
        -1
    } else {
        dp[nums.len() - 1][target as usize]
    }
}

pub fn min_removals(nums: Vec<i32>, target: i32) -> i32 {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert(nums[0], 0);
    if let Some(c) = map.get_mut(&0) {
        *c = (*c).min(1);
    } else {
        map.insert(0, 1);
    }
    for &n in &nums[1..] {
        let mut p = map
            .iter()
            .map(|(&k, &v)| (k ^ n, v))
            .collect::<HashMap<i32, i32>>();
        for (k, v) in map {
            if let Some(c) = p.get_mut(&k) {
                *c = (*c).min(v + 1);
            } else {
                p.insert(k, v + 1);
            }
        }
        map = p;
    }

    if let Some(res) = map.get(&target) {
        *res
    } else {
        -1
    }
}

pub fn find_max_form(strs: Vec<String>, m: i32, n: i32) -> i32 {
    let mut dp = vec![vec![vec![0; n as usize + 1]; m as usize + 1]; strs.len()];
    let (z, o) = cal(strs[0].as_str());
    for j in 0..=m as usize {
        for k in 0..=n as usize {
            if z <= j as usize && o <= k as usize {
                dp[0][j][k] = 1;
            }
        }
    }

    for i in 1..strs.len() {
        let (z, o) = cal(strs[0].as_str());
        for j in 0..=m as usize {
            for k in 0..=n as usize {
                if z <= j as usize && o <= k as usize {
                    dp[i][j][k] = dp[i - 1][j][k].max(dp[i - 1][j - z][k - o] + 1);
                }
            }
        }
    }
    *dp.last().unwrap().last().unwrap().last().unwrap()
}
fn cal(s: &str) -> (usize, usize) {
    (
        s.chars().filter(|&x| x == '0').count(),
        s.chars().filter(|&x| x == '1').count(),
    )
}

pub fn maximum_sale_items(items: Vec<Vec<i32>>, budget: i32) -> i32 {
    let mut copy = vec![0; items.len()];
    for i in 0..items.len() {
        copy[i] = items
            .iter()
            .enumerate()
            .filter(|x| x.0 != i)
            .filter(|x| x.1[0] % items[i][0] == 0)
            .count() as i32
            + 1;
    }
    let mut min = items.iter().map(|x| x[1]).min().unwrap();
    let mut dp = vec![vec![0; budget as usize + 1]; items.len()];
    for j in 0..dp[0].len() {
        if dp[0][j] >= items[0][1] {
            dp[0][j] = copy[0];
        }
    }
    for i in 1..dp.len() {
        for j in 0..dp[0].len() {
            dp[i][j] = dp[i - 1][j];
            if items[i][1] as usize <= j {
                dp[i][j] = dp[i][j].max(dp[i][j - items[i][1] as usize] + copy[i])
            }
        }
    }
    let mut max = 0;
    for (m, value) in dp.last().unwrap().iter().enumerate() {
        let left = budget - m as i32;
        max = max.max(value + left / min);
    }
    max
}

pub fn max_total_reward(mut reward_values: Vec<i32>) -> i32 {
    reward_values.sort();
    let max = reward_values.last().copied().unwrap() as usize;
    let mut dp = vec![vec![false; max * 2]; reward_values.len()];
    for i in 0..dp.len() {
        dp[i][0] = true;
    }
    dp[0][reward_values[0] as usize] = true;
    for i in 1..dp.len() {
        for j in 1..dp[0].len() {
            dp[i][j] = dp[i - 1][j];
            if !dp[i][j] && (j >= reward_values[i] as usize && j < reward_values[i] as usize * 2) {
                dp[i][j] |= dp[i - 1][reward_values[i] as usize - j];
            }
        }
    }
    for (i, x) in dp.last().unwrap().iter().enumerate().rev() {
        if *x {
            return i as i32;
        }
    }
    unreachable!()
}

pub fn min_zero_array(nums: Vec<i32>, queries: Vec<Vec<i32>>) -> i32 {
    let mut res = 0;
    for (i, n) in nums.iter().enumerate() {
        let mut vals = queries
            .iter()
            .map(|x| {
                let l = x[0] as usize;
                let r = x[1] as usize;
                let val = x[2];
                if l <= i && r >= i { val } else { 0 }
            })
            .collect::<Vec<i32>>();
        if *n == 0 {
            continue;
        }
        let mut dp = vec![vec![0; *n as usize + 1]; vals.len() + 1];
        for i in 0..dp.len() {
            dp[i][0] = 1;
        }

        let mut found = false;
        for i in 1..=vals.len() {
            for j in 1..=*n as usize {
                if j as i32 >= vals[i - 1] {
                    dp[i][j] = dp[i - 1][j - vals[i - 1] as usize] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
            if *dp[i].last().unwrap() > 0 {
                found = true;
                res = res.max(i as i32);
                println!("{i}");
                break;
            }
        }
        if !found {
            return -1;
        }
    }
    res
}

pub fn cal_combination(idxs: &[usize], count: usize) -> Vec<Vec<usize>> {
    if count == 0 {
        return vec![];
    }
    let mut res = vec![];

    let mut c = vec![];
    for i in 0..=idxs.len() - count {
        c.push(idxs[i]);
        dfs_cal_conbination(&mut c, idxs, i + 1, count, &mut res);
        c.pop();
    }
    res
}
fn dfs_cal_conbination(
    current: &mut Vec<usize>,
    idx: &[usize],
    start: usize,
    target: usize,
    res: &mut Vec<Vec<usize>>,
) {
    if current.len() == target {
        res.push(current.clone());
    }
    for i in start..idx.len() {
        current.push(idx[i]);
        dfs_cal_conbination(current, idx, i + 1, target, res);
        current.pop();
    }
}

pub fn find_max_path_score(edges: Vec<Vec<i32>>, online: Vec<bool>, k: i64) -> i32 {
    use std::collections::HashMap;
    let mut next_node_map: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
    for e in edges {
        let f = e[0];
        let t = e[1];
        let c = e[2];
        if let None = next_node_map.get_mut(&f).map(|x| x.push((t, c))) {
            next_node_map.insert(f, vec![(t, c)]);
        }
    }
    let mut r = k;
    let mut l = 0;
    while l <= r {
        let mid = (r - l) / 2 + l;
        println!("{mid}");
        if check(mid, &next_node_map, &online, k) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    r as i32
}
fn check(
    mid: i64,
    next_node_map: &std::collections::HashMap<i32, Vec<(i32, i32)>>,
    online: &[bool],
    k: i64,
) -> bool {
    use std::collections::HashSet;
    let mut q = HashSet::new();
    q.insert((0, 0, i32::MAX));
    // let mut q = vec![(0, 0,i32::MAX)];

    while !q.is_empty() {
        let mut next = HashSet::new();
        for (node, total_cost, min_cost) in q {
            // if total_cost > k || !online[node as usize] {
            //     continue;
            // }
            if node == online.len() as i32 - 1 && min_cost as i64 >= mid {
                return true;
            }
            for &(next_node, cost) in next_node_map.get(&node).unwrap_or(&vec![]) {
                if cost as i64 >= mid && cost as i64 + total_cost <= k && online[next_node as usize]
                {
                    next.insert((next_node, cost as i64 + total_cost, min_cost.min(cost)));
                }
            }
        }
        q = next;
    }
    false
}


pub fn transform_str(s: String, strs: Vec<String>) -> Vec<bool> {
    let s_chars: Vec<char> = s.chars().collect();
    let n = s_chars.len();

    // Step 1: Count zeros and ones in original string s
    let s_zeros = s_chars.iter().filter(|&&c| c == '0').count() as i32;
    let s_ones = n as i32 - s_zeros;

    let mut result = Vec::new();

    // Step 2: Iterate through each string in strs
    for target in strs {
        let target_chars: Vec<char> = target.chars().collect();
        let mut target_zeros = 0;
        let mut target_ones = 0;
        let mut question_marks = 0;

        // Count zeros, ones, and question marks in target
        for &c in &target_chars {
            match c {
                '0' => target_zeros += 1,
                '1' => target_ones += 1,
                '?' => question_marks += 1,
                _ => {}
            }
        }
        // println!("{} {} {}", target_zeros, target_ones, question_marks);
        // Step 3: Check if we can fill '?' to match s's counts
        let needed_zeros = s_zeros - target_zeros;
        let needed_ones = s_ones - target_ones;
        // println!("{} {}", needed_zeros, needed_ones);

        // If we need negative numbers or can't fill with available '?'
        if needed_zeros < 0 || needed_ones < 0 || needed_zeros + needed_ones != question_marks {
            result.push(false);
            continue;
        }

        // Step 4: Fill '?' with zeros first (to get minimal lexicographic string)
        let mut filled: Vec<char> = Vec::with_capacity(n);
        let mut remaining_zeros = needed_zeros;
        let mut remaining_ones = needed_ones;

        for &c in &target_chars {
            if c == '?' {
                if remaining_zeros > 0 {
                    filled.push('0');
                    remaining_zeros -= 1;
                } else {
                    filled.push('1');
                    remaining_ones -= 1;
                }
            } else {
                filled.push(c);
            }
        }

        // Step 5: Check if filled string <= s (lexicographically)
        // Because we can reach any string <= s through the sorting operation
        let mut can_transform = true;
        let mut filled_prefix_ones = 0;
        let mut s_prefix_ones = 0;

        for i in 0..n {
            if filled[i] == '1' {
                filled_prefix_ones += 1;
            }
            if s_chars[i] == '1' {
                s_prefix_ones += 1;
            }

            // Check: s must have >= number of 1s in filled for every prefix
            if s_prefix_ones < filled_prefix_ones {
                can_transform = false;
                break;
            }
        }

        result.push(can_transform);
    }

    result
}



pub fn tallest_billboard(rods: Vec<i32>) -> i32 {
    use std::collections::{HashSet};

    let mut max = rods.iter().sum::<i32>() as usize / 2 + 1;
    let mut f = vec![vec![];max];
    f[0].push(HashSet::new());
    for i in 0..rods.len() {
        for j in (rods[i] as usize..max).rev() {
            let mut pre = f[j - rods[i] as usize].clone();
            pre.iter_mut().for_each(|x| {x.insert(i);});
            for x in pre {
                f[j].push(x);
            }
        }
    }
    for i in (0..f.len()).rev() {
        for j in 0..f[i].len() {
            for k in j..f.len() {
                if f[i][j].is_disjoint(&f[i][k]) {
                    return i as i32
                }
            }
        }
    }
    println!("{f:?}");

    0
}
