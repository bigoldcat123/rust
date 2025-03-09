use std::borrow::Cow;

fn main() {
    println!("Hello, world!");
    let a = 100;
    let b = dbg!(a / 100);
    println!("{:?}", b);
    let c = dbg!(do_add(&a, b));
    println!("c{:?}", c);
    let x = &c;
    x.do_add();
    let s = ".".to_string();
    let borrowed_s = Cow::Borrowed(&s);
    let owned_s = borrowed_s.into_owned();
    println!("s{:?}, owned_s:{}", s, owned_s);
}

fn do_add(a: &i32, b: i32) -> i32 {
    *a + b
}
trait Ext {
    fn do_add(&self) {}
}
impl Ext for i32 {
    fn do_add(&self) {
        println!("self: {:?}", self);
    }
}
