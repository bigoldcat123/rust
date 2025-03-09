pub fn is_armstrong_number(num: u32) -> bool {
    let num_str = num.to_string();
    let p = num_str.len() as u32;
    let res = num_str
        .chars()
        .map(|x| x.to_digit(10).unwrap())
        .map(|x| x.pow(p))
        .sum::<u32>();
    res == num
}
