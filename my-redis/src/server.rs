use std::{collections::HashMap, pin::Pin, sync::Arc, vec};

use tokio::{
    io::{self, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::Mutex,
};

#[derive(Debug)]
enum Cmd {
    GET(String),
    SET(String, String),
}
impl TryFrom<&Vec<String>> for Cmd {
    type Error = String;
    fn try_from(value: &Vec<String>) -> Result<Self, Self::Error> {
        if let Some(cmd) = value.get(0) {
            match cmd.to_lowercase().as_str() {
                "get" => {
                    if let Some(v) = value.get(1) {
                        return Ok(Cmd::GET(v.into()));
                    } else {
                        return Err("get <key>".into());
                    }
                }
                "set" => {
                    if value.len() == 3 {
                        return Ok(Cmd::SET(
                            value.get(1).unwrap().into(),
                            value.get(2).unwrap().into(),
                        ));
                    } else {
                        return Err("set <key> <value>".into());
                    }
                }
                _ => {
                    return Err(format!("{} is not supported", cmd));
                }
            }
        } else {
            return Err("input something!".into());
        }
    }
}
impl Cmd {
    async fn handle_cmd<T: DataCenter<String, String>>(
        self,
        mut writer: Pin<&mut impl AsyncWrite>,
        data_center: Arc<Mutex<T>>,
    ) {
        match self {
            Cmd::GET(key) => {
                let data_center = data_center.lock().await;
                let value = data_center
                    .get(key)
                    .or_else(|err| Ok::<String, String>(err))
                    .unwrap();
                writer.write_all(value.as_bytes()).await.unwrap();
            }
            Cmd::SET(key, value) => {
                let mut data_center = data_center.lock().await;
                data_center.set(key, value);
                writer.write_all(b"ok").await.unwrap();
            }
        }
    }
    async fn exec<T: DataCenter<String, String>>(
        bytes: &[u8],
        mut writer: Pin<&mut impl AsyncWrite>,
        data_center: Arc<Mutex<T>>,
    ) -> ExecStat {
        let bytes = bytes.trim_ascii_start();
        let mut args = vec![];
        let mut buf = vec![];

        for (idx, ele) in bytes.iter().enumerate() {
            if *ele == b'\n' {
                if buf.len() != 0 {
                    if let Ok(s) = String::from_utf8(buf.clone()) {
                        args.push(s);
                        buf.clear();
                    } else {
                        return ExecStat::ERROR(idx + 1);
                    }
                }
                match Cmd::try_from(&args) {
                    Ok(cmd) => {
                        cmd.handle_cmd(writer, data_center).await;
                        return ExecStat::OK(idx + 1);
                    }
                    Err(msg) => {
                        writer.write_all(msg.as_bytes()).await.unwrap_or_default();
                        return ExecStat::ERROR(idx + 1);
                    }
                }
            }

            if *ele != b' ' {
                buf.push(*ele);
            } else if bytes[idx - 1] != b' ' {
                if let Ok(s) = String::from_utf8(buf.clone()) {
                    args.push(s);
                    buf.clear();
                } else {
                    return ExecStat::ERROR(idx);
                }
            }
        }
        return ExecStat::LACK;
    }
}

#[derive(Debug)]
enum ExecStat {
    OK(usize),
    LACK,
    ERROR(usize),
}
struct Client<T: DataCenter<String, String>> {
    inner: TcpStream,
    buffer: Vec<u8>,
    cursor: usize,
    data_center: Arc<Mutex<T>>,
}

impl<T: DataCenter<String, String>> Client<T> {
    async fn process(mut self) -> io::Result<()> {
        let (mut rh, mut wh) = self.inner.split();

        loop {
            match Cmd::exec(
                &self.buffer[..self.cursor],
                std::pin::pin!(&mut wh),
                Arc::clone(&self.data_center),
            )
            .await
            {
                ExecStat::OK(used_size) => {
                    self.cursor = 0;
                    self.buffer.drain(..used_size);
                }
                ExecStat::LACK => {
                    let size = rh.read(&mut self.buffer[self.cursor..]).await?;

                    if size == 0 {
                        eprintln!("peer connection broken !");
                        break;
                    }
                    self.cursor += size;
                }
                ExecStat::ERROR(used_size) => {
                    // wh.write(b"somethingWrong").await?;
                    self.cursor = 0;
                    self.buffer.drain(..used_size);
                }
            }
        }
        Ok(())
    }
}

impl<T: DataCenter<String, String>> From<(TcpStream, Arc<Mutex<T>>)> for Client<T> {
    fn from(value: (TcpStream, Arc<Mutex<T>>)) -> Self {
        Self {
            inner: value.0,
            data_center: value.1,
            buffer: vec![0; 4096],
            cursor: 0,
        }
    }
}

pub struct FileCheckerServer {
    inner: TcpListener,
}

impl FileCheckerServer {
    pub async fn new(addr: impl ToSocketAddrs) -> io::Result<Self> {
        let inner = TcpListener::bind(addr).await?;
        Ok(Self { inner })
    }
    pub async fn run(self) -> io::Result<()> {
        let data_center = MemoryDataCenter::new();
        let data_center = Arc::new(Mutex::new(data_center));
        while let Ok((socket, _)) = self.inner.accept().await {
            let c = Client::from((socket, Arc::clone(&data_center)));

            tokio::spawn(async move { c.process().await });
        }
        Ok(())
    }
}

pub trait DataCenter<K, V> {
    fn set(&mut self, key: K, value: V);
    fn get(&self, key: K) -> Result<V, String>;
}

// #[derive(Send)]
struct MemoryDataCenter {
    data: HashMap<String, String>,
}

impl MemoryDataCenter {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl DataCenter<String, String> for MemoryDataCenter {
    fn set(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }
    fn get(&self, key: String) -> Result<String, String> {
        match self.data.get(&key) {
            None => Err(format!("{} do not exist!", key)),
            Some(value) => Ok(value.into()),
        }
    }
}
