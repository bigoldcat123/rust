
fn main() {
    let s = Str { value: "".into() };
    // unsafe { s }
    let x = &s;
    let x2 = s.as_ref();
    println!("{:?}", x.value);
    println!("{:?}", x2);

    let mut a: Option<String> = Some("()".into());

    let xx = a.as_mut().unwrap();
    xx.push_str("string");
    println!("{:?}", a);
    let mut a = Box::new("".to_string());
    let _s = a.as_ref();
    let _s = a.as_mut();
    // pin::Pin::new(pointer)
}

struct Str {
    value: String,
}

impl AsRef<String> for Str {
    fn as_ref(&self) -> &String {
        &self.value
    }
}

// impl Into<String> for Str {
//     fn into(self) -> String {
//         "".to_string()
//     }
// }
// impl From<Str> for String {
//     fn from(value: Str) -> Self {
//         "".to_string()
//     }
// }
