use std::{path::PathBuf, str::FromStr};

use http::Request;
use http_play::{ReqSend, enforce::EnforceBody, formdata_enforce::FormDataAdaptor};
use http_play::builder::Person;

#[test]
fn test_use() {
    let mut form_data_adaptor = FormDataAdaptor::new();
    form_data_adaptor.write_field("hello", "fuckyoaaau");
    form_data_adaptor.write_field("hello2", "2fucky????????ou");
    form_data_adaptor.write_path(
        "fuckyou",
        PathBuf::from_str("/Users/dadigua/Desktop/keyboard-shortcuts-macos 2.pdf").unwrap(),
        "image/png",
    );
    form_data_adaptor.write_path(
        "fuckyou2",
        PathBuf::from_str("/Users/dadigua/Desktop/DCIM/2025-20/_MG_1953.CR3").unwrap(),
        "image/png",
    );
    form_data_adaptor.write_field("hello3", "world!");

    let e = Request::post("https://localhost:3001/hello/asdjl")
        .header("Host", "localhost:3000")
        .enforce_body(form_data_adaptor)
        .unwrap();
    let _ = e.send::<String>();
}


#[test]
pub fn t () {
    let person = Person::builder()
        .name("??e")
        .address("eeeee")
        .age(19u8)
        .build();
    println!("{}",person);
}