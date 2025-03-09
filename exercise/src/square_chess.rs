pub fn square(s: u32) -> u64 {
    // todo!("grains of rice on square {s}");
    match s {
        1 => 1,
        2 => 2,
        _ => 2_u64.pow(s-1),
    }
}
//1 2 4 8 16 32 64 128
pub fn total() -> u64 {
    let mut res = 0;
    for i in 1..=64_u32 {
        res += square(i);
    }

    res
}
