const TAG_LEN: usize = 5;
const TAG: [&str; TAG_LEN] = ["酸", "甜", "苦", "辣", "咸"];

fn get_food_tag() -> Vec<f32> {
    let mut ans = vec![0.0; TAG_LEN];
    if rand::random_bool(1.0 / 2.0) {
        ans[0] = 1.0;
    }
    if rand::random_bool(1.0 / 2.0) {
        ans[1] = 1.0;
    }
    if rand::random_bool(1.0 / 2.0) {
        ans[2] = 1.0;
    }
    if rand::random_bool(1.0 / 2.0) {
        ans[3] = 1.0;
    }
    if rand::random_bool(1.0 / 2.0) {
        ans[4] = 1.0;
    }
    ans
}

fn get_food_tag_random() -> Vec<f32> {
    let mut ans = vec![0.0; TAG_LEN];
    for i in 0..TAG_LEN {
        ans[i] = if rand::random_bool(1.0 / 2.0) {
            1.0
        } else {
            0.0
        };
    }
    ans
}

fn cal_user_profile(foods: &[Vec<f32>]) -> Vec<f32> {
    let mut profile = vec![0.0; TAG_LEN];
    let mut sum = 0.0;
    for food in foods {
        for (i, &tag) in food.iter().enumerate() {
            profile[i] += tag;
            sum += tag;
        }
    }
    println!("{:?}", profile);
    for i in 0..TAG_LEN {
        profile[i] /= sum;
    }
    profile
}

fn test_food_similarity(food: &[f32], profile: &[f32]) -> f32 {
    let dot = food.iter().zip(profile).map(|(a, b)| a * b).sum::<f32>();
    let norm = food.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_profile = profile.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if dot == 0.0 || norm == 0.0 || norm_profile == 0.0 {
        return 0.0;
    }
    dot / (norm * norm_profile)
}

#[test]
fn feature() {
    let mut foods = vec![];
    for _ in 0..10000 {
        foods.push(get_food_tag());
    }
    let profile = cal_user_profile(&foods);
    println!("{:?}", profile);
    for _ in 0..10 {
        let food = get_food_tag_random();

        println!("{food:?} -> {:?}", test_food_similarity(&food, &profile));
    }
}
