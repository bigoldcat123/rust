use std::str::FromStr;

use http::{request::Builder, HeaderName, HeaderValue, Request, Version};

use crate::RequestBodyAdaptor;

pub trait AsStr {
    fn as_str(&self) -> &str;
}

impl AsStr for Version {
    fn as_str(&self) -> &str {
        match *self {
            Version::HTTP_09 => "HTTP/0.9",
            Version::HTTP_10 => "HTTP/1.0",
            Version::HTTP_11 => "HTTP/1.1",
            Version::HTTP_2 => "HTTP/2.0",
            Version::HTTP_3 => "HTTP/3.0",
            _ => unreachable!(),
        }
    }
}

pub trait EnforceBody {
    fn enforce_body<T: RequestBodyAdaptor>(self, body: T) -> http::Result<Request<T>>;
}

impl EnforceBody for Builder {
    ///
    /// ## enhance body
    /// this will add related headers automatically
    /// 
    /// Added headers:
    /// 1. Content-Type
    /// 2. Content-Length
    fn enforce_body<T: RequestBodyAdaptor>(mut self, body: T) ->  http::Result<Request<T>> {
        let e = self.headers_mut().unwrap();
        for (k, v) in body.default_headers() {
            e.insert(
                HeaderName::from_str(&k).unwrap(),
                HeaderValue::from_str(&v).unwrap(),
            );
        }
        self.body(body)
    }
}