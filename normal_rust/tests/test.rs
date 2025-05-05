use std::collections::HashMap;

#[test]
fn life_time() {
    struct NumberIter<'a> {
        nums: &'a [u8],
    }
    impl<'a> NumberIter<'a> {
        fn next(&mut self) -> Option<&'a u8> {
            if self.nums.is_empty() {
                None
            } else {
                let e = &self.nums[0];
                self.nums = &self.nums[1..];
                Some(e)
            }
        }
    }
    let mut iter: NumberIter<'_> = NumberIter { nums: &[1, 1, 3] };
    let a = iter.next();
    let b = iter.next();
    assert_eq!(a, b);
    use std::sync::Mutex;

    struct Struct {
        mutex: Mutex<String>,
    }

    impl Struct {
        // downgrades mut self to shared str
        fn get_string(&mut self) -> &str {
            self.mutex.get_mut().unwrap()
        }
        fn mutate_string(&self) {
            // if Rust allowed downgrading mut refs to shared refs
            // then the following line would invalidate any shared
            // refs returned from the get_string method
            *self.mutex.lock().unwrap() = "surprise!".to_owned();
        }
    }

    let mut map = HashMap::new();
    let s = "adasdasd";
    for c in s.chars() {
        map.entry(c).and_modify(|x| *x += 1).or_insert(1);
    }
    println!("{:?}", map);

    let identity: &dyn Fn(&i32) -> &i32 = &|x: &i32| x;
    identity(&2);

    fn fn_take_a<'a, T: 'a>(t: T) {}
    let a = 1;
    let b = String::new();
    let c = "";
    fn_take_a(a);
    fn_take_a(b);
    fn_take_a(c);
}
