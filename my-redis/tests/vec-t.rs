#[test]
fn test_function() {
    let mut a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    a.drain(..2);
    println!("{:?}", a);
}
