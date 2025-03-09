fn num2str(num: u32) -> String {
    match num {
        1 => "One".to_string(),
        2 => "Two".to_string(),
        3 => "Three".to_string(),
        4 => "Four".to_string(),
        5 => "Five".to_string(),
        6 => "Six".to_string(),
        7 => "Seven".to_string(),
        8 => "Eight".to_string(),
        9 => "Nine".to_string(),
        10 => "Ten".to_string(),
        _ => "No".to_string(),
    }
}

fn get_bottle(num:u32) -> &'static str {
    if num > 1 || num == 0 {
         "bottles"
    }else {
        "bottle"
    }
}
pub fn recite(start_bottles: u32, take_down: u32) -> String {
    let mut res = String::new();

    for idx in { (start_bottles - take_down + 1)..=start_bottles }.rev() {
        let line = format!("{} green {} hanging on the wall,\n", num2str(idx),get_bottle(idx));
        res.push_str(&line.repeat(2));
        res.push_str("And if one green bottle should accidentally fall,\n");
        res.push_str(&format!(
            "There'll be {} green {} hanging on the wall.\n\n",
            num2str(idx - 1).to_lowercase(),
            get_bottle(idx - 1)
        ));
    }
    println!("{}",res);
    res
}
