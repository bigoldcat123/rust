use std::{
    io::{Read, Write},
    net::TcpStream,
};

use enforce::AsStr;
use http::{Request, Response};

pub mod enforce;
pub mod formdata;
pub mod formdata_enforce;
pub mod builder;

pub trait RequestBodyAdaptor {
    fn write(&mut self, writer: impl Write);
    fn default_headers(&self) -> Vec<(String, String)>;
}

pub trait ReqSend {
    fn send<R>(self) -> Result<Response<R>, String>;
}
impl<T: RequestBodyAdaptor> ReqSend for Request<T> {
    fn send<R>(self) -> Result<Response<R>, String> {
        let mut client = TcpStream::connect(format!(
            "{}:{}",
            self.uri().host().unwrap(),
            self.uri().port().unwrap()
        ))
        .unwrap();

        send_(self, &mut client);

        let mut buf = vec![];
        client.read_to_end(&mut buf).unwrap();
        println!("{}", String::from_utf8_lossy(&buf));
        Err("()".to_string())
    }
}

fn send_<T: RequestBodyAdaptor>(mut req: Request<T>, mut writer: impl Write) {
    write!(
        &mut writer,
        "{} {} {}\r\n",
        req.method().as_str(),
        req.uri().path(),
        req.version().as_str()
    )
    .unwrap();
    for (name, value) in req.headers() {
        write!(&mut writer, "{}: {}\r\n", name, value.to_str().unwrap()).unwrap();
    }
    write!(&mut writer, "\r\n").unwrap();
    req.body_mut().write(writer);
}
