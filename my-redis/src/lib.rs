use std::{
    env,
    io::{self, Error, Read, Write, stdin, stdout},
    net::{TcpStream, ToSocketAddrs},
    process,
};
pub mod server;
// enum Frame {
//     COMMAND(String),
//     ARG(String),
// }

pub struct FileChecker {
    inner: TcpStream,
}

impl FileChecker {
    pub fn new(addr: impl ToSocketAddrs) -> Result<Self, Error> {
        let inner = TcpStream::connect(addr)?;
        Ok(Self { inner })
    }
    pub fn run(mut self) -> io::Result<()> {
        let mut line = String::new();
        let mut read_buf = vec![0; 128];
        loop {
            indicate()?;
            line.clear();
            stdin().read_line(&mut line)?;
            if line.starts_with("exit") {
                return Ok(());
            }
            if line.trim().len() == 0 {
                continue;
            }
            self.inner.write_all(line.as_bytes())?;
            self.inner.flush().unwrap();
            let r = self.inner.read(&mut read_buf).unwrap();
            println!(
                "res {:?}",
                String::from_utf8(read_buf[..r].to_vec()).unwrap()
            );
        }
    }
}

fn indicate() -> io::Result<()> {
    print!("> ");
    stdout().flush()?;
    Ok(())
}

pub fn parse_args() -> impl ToSocketAddrs {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("usage client <host> <port>");
        process::exit(1);
    };
    format!("{}:{}", &args[1], &args[2])
}
