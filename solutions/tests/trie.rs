use std::{collections::{HashMap, HashSet}, fs::File, io::Read, path::Path, time::SystemTime};


struct Trie {
    is_word:bool,
    next:HashMap<char,Self>
}
impl Trie {
    fn new() -> Self {
        Self { is_word: false, next: HashMap::new() }
    }
    fn insert(&mut self,s:&str) {
        let mut node = self;
        for c in s.chars() {
            if node.next.get(&c).is_none() {
                node.next.insert(c, Self::new());
            }
            node = node.next.get_mut(&c).unwrap();
        }
        node.is_word = true
    }
    fn search(&self,s:&str) -> bool {
        let mut node = self;
        for c in s.chars() {
            if node.next.get(&c).is_none() {
                return false
            }
            node = node.next.get(&c).unwrap();
        }
        node.is_word
    }
}

fn get_text<P: AsRef<Path>>(path:P) -> Vec<String> {
    let mut f = File::open(path).unwrap();
    let mut res = String::new();
    f.read_to_string(&mut res).unwrap();
    res.split_whitespace().map(|x| x.to_string()).collect()
}

#[test]
fn test_trie() {
    let mut t = Trie::new();
    let mut set = HashSet::new();
    let texts = get_text("/Users/dadigua/Desktop/edu practice 2025 3-7/code/rust/solutions/src/five.rs");
    let now = SystemTime::now();

    for text in &texts {
        t.insert(text);
    }

    println!("Trie: {:?}",now.elapsed().unwrap());

    let now = SystemTime::now();

    for text in &texts {
        set.insert(text);
    }

    println!("Set: {:?}",now.elapsed().unwrap());
    // search
    println!("search");
    // let  texts = get_text("/Users/dadigua/Desktop/edu practice 2025 3-7/code/rust/solutions/src/four.rs");
    let now = SystemTime::now();
    for text in &texts {
        t.search(text);
    }
    println!("Trie: {:?}",now.elapsed().unwrap());
    let now = SystemTime::now();
    for text in &texts {
        set.contains(text);
    }
    println!("Set: {:?}",now.elapsed().unwrap());


}
