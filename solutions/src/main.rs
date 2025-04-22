use log::{debug, error, log_enabled};
use solutions::A;

fn main() {
    env_logger::init();
    debug!("hello-{}","dalaomao");

    if log_enabled!(log::Level::Error) {
        error!("oh no!")
    }
    A::log_play();
}
