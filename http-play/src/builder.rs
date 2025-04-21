use std::fmt::{Display, Formatter};

pub struct Person {
    name: String,
    age: u8,
    address: String,
    number:String
}

impl Person {
    pub fn builder() -> Builder {
        Builder {
            inner:Person::default()
        }
    }
}

impl Display for Person {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f,"Name: {}, Age: {}, Address: {}", self.name,self.age,self.address)
    }
}
impl Default for Person {
    fn default() -> Self {
        Self {
            name:String::default(),
            age: u8::default(),
            address:String::default(),
            number:String::default()
        }
    }
}

pub struct Builder {
    inner: Person,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            inner: Person::default(),
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.inner.name = name.to_string();
        self
    }
    pub fn age(mut self, age: u8) -> Self {
        self.inner.age = age;
        self
    }
    pub fn address(mut self, address: &str) -> Self {
        self.inner.address = address.to_string();
        self
    }
    pub fn number(mut self, number: &str) -> Self {
        self.inner.number = number.to_string();
        self
    }
    pub fn build(self) -> Person {
        let e = self.inner;
        e
    }
}

