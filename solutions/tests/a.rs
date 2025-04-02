use solutions::A;

#[test]
fn test_function() {
    let e = A::generate(5);
    println!("{:#?}",e);
}

#[test]
fn ele() {
    let a = ["aa","b"];
    let s = "a".to_string();
    println!("{:?}",a.contains(&&s[..]));
}