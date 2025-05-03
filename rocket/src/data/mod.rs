pub mod state;
use rocket::{
    FromForm, Request, State,
    form::{self, Error},
    http::Status,
    request::{self, FromRequest, Outcome},
    serde::{Deserialize, Serialize},
};
use state::Counter;

#[derive(FromForm)]
pub struct User<'a> {
    pub name: &'a str,
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(crate = "rocket::serde")]
pub struct Stu<'a> {
    pub name: &'a str,
}

#[derive(FromForm, Debug)]
pub struct Task<'a> {
    #[field(validate=eq(true))]
    pub complete: bool,
    #[field(validate = is_czh( "name"))]
    pub r#type: &'a str,
}

fn is_czh<'t>(s: &'t str, name: &'static str) -> form::Result<'t, ()> {
    if s == name {
        Ok(())
    } else {
        Err(Error::validation("invalid credit card number"))?
    }
}

pub struct ApiKey<'k> {
    pub key: &'k str,
}

#[rocket::async_trait]
impl<'r> FromRequest<'r> for ApiKey<'r> {
    type Error = String;

    async fn from_request(req: &'r Request<'_>) -> request::Outcome<Self, Self::Error> {
        let data: rocket::outcome::Outcome<&State<Counter>, (Status, ()), Status> =
            req.guard::<&State<Counter>>().await;
            
        if let rocket::outcome::Outcome::Success(e) = data {
            e.inner.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        if let Some(key) = req.headers().get_one("api-key") {
            Outcome::Success(ApiKey { key })
        } else {
            Outcome::Error((Status::Forbidden, format!("Wrong!")))
        }
    }
}
