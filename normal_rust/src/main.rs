use normal_rust::hello;

static mut C: i32 = 100;
fn main() {
    let args  = std::env::args().collect::<Vec<String>>();
    println!("{:?}",args);
    const X: i32 = 10;
    let a = 1;
    let b = 2;
    let c = a + b;
    unsafe {
        C = 1;
        println!("{:?}", c + X + C);
    }
    hello();
}


#[cfg(test)]
mod test {
    use crate::hello;

    #[test]
    fn test_function() {
        hello();
    }
}
