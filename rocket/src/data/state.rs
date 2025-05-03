use std::{
    sync::{Arc, atomic::AtomicI32},
    time::Duration,
};

use log::{error, info};
use rocket::futures::lock::Mutex;
use rppal::gpio::{self, Gpio};

pub struct Counter {
    pub inner: AtomicI32,
}

pub struct Light {
    pin_out: Arc<Mutex<gpio::OutputPin>>,
    pin_ipt: gpio::InputPin,
}

impl Light {
    pub fn new(pin_out: u8, pin_ipt: u8) -> Result<Self, gpio::Error> {
        let out = gpio::Gpio::new()?.get(pin_out)?.into_output();
        let out = Arc::new(Mutex::new(out));
        let mut ipt = Gpio::new()?.get(pin_ipt)?.into_input_pulldown();
        let out_clone = Arc::clone(&out);
        ipt.set_async_interrupt(
            gpio::Trigger::RisingEdge,
            Some(Duration::from_millis(50)),
            move |_| {
                info!("interrupt!");
                if let Some(mut l) = out_clone.try_lock() {
                    if l.is_set_high() {
                        l.set_low();
                    } else {
                        l.set_high();
                    }
                } else {
                    error!("try lock error");
                }
            },
        )?;
        Ok(Self {
            pin_out: out,
            pin_ipt: ipt,
        })
    }
    pub fn toggle(&mut self) {
        if let Some(mut l) = self.pin_out.try_lock() {
            if l.is_set_high() {
                l.set_low();
            } else {
                l.set_high();
            }
        } else {
            error!("error! to get lock!");
        }
    }

    pub fn state(&self) -> bool {
        if let Some(l) = self.pin_out.try_lock() {
            l.is_set_high()
        } else {
            error!("error! to get lock!");
            false
        }
    }
}
