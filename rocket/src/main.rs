#[macro_use]
extern crate rocket;

use std::{
    env,
    net::Ipv4Addr,
    path::{Path, PathBuf},
    sync::{Mutex, atomic::AtomicI32},
};

use log::info;
use rocket::{
    Config, Data, State,
    data::ByteUnit,
    form::Form,
    fs::{FileServer, NamedFile, TempFile},
    http::CookieJar,
    serde::json::Json,
    tokio,
};
use rocket_study::data::{
    ApiKey, Stu, Task, User,
    state::{Counter, Light},
};
#[get("/<file..>")]
async fn home(file: PathBuf) -> NamedFile {
    NamedFile::open(Path::new("/Users/dadigua/Pictures").join(file))
        .await
        .unwrap()
}

#[post("/", data = "<user>")]
async fn enroll(user: Form<User<'_>>) -> Option<&'_ str> {
    let u = user.into_inner();
    println!("{:?}", u.name);
    if u.name.len() > 10 {
        Some(u.name)
    } else {
        None
    }
}
#[catch(404)]
fn not_found() -> String {
    format!("{}", "e")
}

#[get("/user/<id>")]
fn user(id: Result<usize, &str>) {
    /* ... */
    if let Ok(e) = id {
        info!("{}", e);
    } else if let Err(er) = id {
        error!("{}", er)
    }
}

#[get("/user/<id>", rank = 2)]
fn user_int(id: isize) {
    /* ... */
    info!("{}", id);
}

#[get("/user/<id>", rank = 3)]
fn user_str(id: &str) {
    /* ... */
    info!("{}", id);
}

#[get("/cookie")]
fn try_cookie(cookie: &CookieJar) -> Option<String> {
    cookie
        .get("hello")
        .map(|x| format!("the hello is ~{}", x.value()))
}

#[post("/stu", format = "json", data = "<stu>")]
pub fn stu_json(stu: Json<Stu<'_>>) {
    let e = stu.into_inner();
    println!("{:?}", e);
}

#[post("/upload", format = "plain", data = "<file>")]
async fn upload(file: TempFile<'_>) -> std::io::Result<()> {
    let f = file.path().unwrap();
    info!("{:?}", f);
    info!("{}", file.len());
    Ok(())
}
#[catch(400)]
fn catch_400() -> &'static str {
    "400..."
}

#[catch(403)]
fn catch_403() -> &'static str {
    "yet you are not allowed!"
}

#[post("/stream", data = "<file>")]
async fn data(file: Data<'_>) -> std::io::Result<()> {
    file.open(ByteUnit::Byte(10))
        .stream_to(tokio::io::stdout())
        .await?;
    Ok(())
}

#[post("/form", data = "<task>")]
async fn form(task: Form<Task<'_>>) {
    info!("{:?}", task);
}
#[get("/json?<name>")]
fn json(name: &'_ str) -> Json<Stu<'_>> {
    info!("{}", name);
    Json(Stu { name: "hello" })
}

#[get("/api")]
fn api_key(key: ApiKey) -> &'static str {
    info!("the key is {}", key.key);
    "yes you are ok!"
}

#[get("/count")]
fn count(counter: &State<Counter>) -> String {
    counter
        .inner
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!(
        "current Counter is {:?}",
        counter.inner.load(std::sync::atomic::Ordering::Relaxed)
    )
}

#[get("/status")]
pub fn led_status(light: &State<Mutex<Light>>) -> &'static str {
    if let Ok(e) = light.lock() {
        if e.state() { "light!" } else { "dark!" }
    } else {
        "sorry~"
    }
}
#[get("/toggle")]
pub fn led_toggle(light: &State<Mutex<Light>>) -> &'static str {
    if let Ok(mut e) = light.lock() {
        e.toggle();
        "ok!"
    } else {
        "sorry~"
    }
}
#[launch]
fn launch() -> _ {
    // env_logger::init();
    let s = env::args().collect::<Vec<String>>();
    debug!("{:?}", s);
    println!("{:?}", s);
    rocket::Rocket::build()
        .manage(Counter {
            inner: AtomicI32::new(0),
        })
        .manage(Mutex::new(
            Light::new(23, 24).expect("wrong with init light!"),
        ))
        // .mount(
        //     "/",
        //     FileServer::from("/Users/dadigua/Desktop/edu practice 2025 3-7/期中考试-上机"),
        // )
        .mount("/led", routes![led_toggle, led_status])
        .mount("/api", routes![enroll, home])
        .mount(
            "/test",
            routes![
                user, user_int, user_str, try_cookie, stu_json, upload, data, form, json, api_key,
                count
            ],
        )
        .register("/", catchers![not_found, catch_400, catch_403])
        .configure(Config {
            address: std::net::IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            log_level:rocket::config::LogLevel::Normal,
            ..Config::debug_default()
        })
}
