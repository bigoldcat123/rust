use std::{collections::HashSet, io::SeekFrom};

struct TireSolutions;
impl TireSolutions {
    fn word_dictionary_impl() {
        use std::collections::HashMap;
        #[derive(Default)]
        struct WordDictionary {
            is_word: bool,
            next: HashMap<char, WordDictionary>,
        }
        impl WordDictionary {
            fn new() -> Self {
                Default::default()
            }

            fn add_word(&mut self, word: String) {
                let mut node = self;
                for c in word.chars() {
                    if node.next.get(&c).is_none() {
                        node.next.insert(c, Default::default());
                    }
                    node = node.next.get_mut(&c).unwrap();
                }
                node.is_word = true;
            }

            fn search(&self, word: String) -> bool {
                self.dfs_search(&word.chars().collect::<Vec<char>>(), 0)
            }
            fn dfs_search(&self, word: &[char], current: usize) -> bool {
                if current == word.len() && self.is_word {
                    return true;
                } else {
                    return false;
                }
                if word[current] == '.' {
                    for (k, v) in self.next.iter() {
                        if v.dfs_search(word, current + 1) {
                            return true;
                        }
                    }
                } else {
                    if let Some(n) = self.next.get(&word[current]) {
                        return n.dfs_search(word, current + 1);
                    }
                };
                false
            }
        }
    }
    fn word_search() {
        use std::collections::{HashMap, HashSet};
        #[derive(Clone)]
        struct Tire {
            next: [Option<Box<Tire>>; 26],
        }
        impl Tire {
            fn new() -> Self {
                Self {
                    next: [
                        None, None, None, None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None, None, None, None, None, None, None, None,
                        None, None,
                    ],
                }
            }
            fn build(
                &mut self,
                board: &Vec<Vec<char>>,
                i: i32,
                j: i32,
                visited: &mut HashSet<(i32, i32)>,
                len: usize,
            ) {
                if i < 0
                    || i as usize >= board.len()
                    || j < 0
                    || j as usize >= board[0].len()
                    || visited.contains(&(i, j))
                    || len >= 10
                {
                    return;
                }
                let i_usze = i as usize;
                let j_usize = j as usize;
                let idx = (board[i_usze][j_usize] as u8 - 97) as usize;
                if self.next[idx].is_none() {
                    self.next[idx] = Some(Box::new(Self::new()));
                }
                let next = self.next[idx].as_mut().unwrap();
                visited.insert((i, j));
                let next_steps = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)];
                for n in next_steps {
                    next.build(board, n.0, n.1, visited, len + 1);
                }
                visited.remove(&(i, j));
            }
            fn search(&self, s: &str) -> bool {
                let mut h = self;
                for c in s.chars() {
                    let idx = (c as u8 - 97) as usize;
                    if h.next[idx].is_none() {
                        return false;
                    }
                    h = h.next[idx].as_ref().unwrap();
                }
                true
            }
        }
        pub fn find_words(board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
            let mut res = vec![];
            let mut t = Tire::new();
            let mut visited = HashSet::new();
            for i in 0..board.len() {
                for j in 0..board[0].len() {
                    visited.clear();
                    t.build(&board, i as i32, j as i32, &mut visited, 0);
                }
            }
            for w in words {
                if t.search(&w) {
                    res.push(w);
                }
            }
            res
        }
    }
    pub fn count_prefix_suffix_pairs(mut words: Vec<String>) -> i64 {
        use std::collections::HashMap;
        struct Trie {
            count:i64,
            next:HashMap<(u8,u8),Trie>
        }
        impl Trie {
            fn new() -> Self {
                Self{
                    count:0,
                    next:HashMap::new()
                }
            }
            fn insert(&mut self,s:&str) {
                let mut node = self;
                let s = s.as_bytes();
                let mut l = 0;
                let mut r = s.len() as i32 - 1;
                while r >= 0 {
                    let key = (s[l],s[r as usize]);
                    if node.next.get(&key).is_none() {
                        node.next.insert(key, Self::new());
                    }
                    node = node.next.get_mut(&key).unwrap();
                    node.count += 1;
                    l += 1;
                    r -= 1;
                }
            }
            fn search(&self,s:&str) -> i64 {
                let s = s.as_bytes();
                let mut l = 0;
                let mut r = s.len() as i32 - 1;
                let mut node = self;
                while r >= 0 {
                    let key = (s[l],s[r as usize]);
                    if node.next.get(&key).is_none() {
                        return 0
                    }
                    node = node.next.get(&key).unwrap();
                    l += 1;
                    r -= 1;
                }
                node.count
            }
        }

        let mut t = Trie::new();
        let mut res = 0;
        while let Some(word) = words.pop() {
            res += t.search(&word);
            t.insert(&word);
        }
        res
    }
    pub fn word_break(s: String, word_dict: Vec<String>) -> bool {
        use std::collections::HashSet;
        let set:HashSet<String> = HashSet::from_iter(word_dict.into_iter());
        let mut dp = vec![false;s.len()];
        for i in 1..=s.len() {
            if set.contains(&s[0..i]) {
                dp[i - 1] = true;
            }else {
                for j in (1..=i - 1).rev() {
                    if dp[j - 1] && set.contains(&s[j..i]){
                        dp[i - 1] = true;
                        break;
                    }
                }
            }
        }
        dp.last().copied().unwrap()
    }
    fn aaa() {
        let mut a = Some(Box::new(String::from("value")));
        let x = a.as_mut().unwrap().as_mut();

    }
}
