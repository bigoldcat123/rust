pub mod armstring;
pub mod bottole_song;
pub mod luhn;
pub mod square;

pub mod square_chess;

pub mod acrony {
    pub fn abbreviate(phrase: &str) -> String {
        let mut res = String::new();
        phrase.split_whitespace().for_each(|row_word| {
            row_word.split("-").for_each(|word| {
                if word.to_lowercase() == word {
                    res.push(word.chars().next().unwrap());
                    return;
                }
                let pre = None::<char>;
                for ele in word.chars() {
                    if ele.is_uppercase() {
                        res.push(ele);
                    }else {
                    }
                }
            });
        });
        res
    }
}
