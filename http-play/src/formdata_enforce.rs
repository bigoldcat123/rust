use std::{collections::HashMap, ffi::OsStr, fs::File, io::Write, path::PathBuf, time::SystemTime};

use base64::Engine;
use rand::{RngCore, thread_rng};

use crate::{RequestBodyAdaptor, formdata::FormData};

enum Field {
    Str(String),
    File(PathBuf, String),
}

pub struct FormDataAdaptor {
    fields: HashMap<String, Field>,
    len: usize,
    boundary: String,
}

impl FormDataAdaptor {
    pub fn new() -> Self {
        let mut buf = [0; 24];

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("system time should be after the Unix epoch");
        (&mut buf[..4]).copy_from_slice(&now.subsec_nanos().to_ne_bytes());
        (&mut buf[4..12]).copy_from_slice(&now.as_secs().to_ne_bytes());
        thread_rng().fill_bytes(&mut buf[12..]);

        let boundary = format!(
            "{:->68}",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&buf)
        );
        println!("{:?}", boundary.len());
        let len = boundary.len() + 2 + 2 + 2;
        Self {
            fields: HashMap::new(),
            len,
            boundary,
        }
    }
    pub fn write_field(&mut self, name: &str, value: &str) {
        self.fields
            .insert(name.to_string(), Field::Str(value.to_string()));
        let len = Self::get_header_len(name, None, None, &self.boundary);
        self.len += len;
        self.len += value.len() + 2;
    }
    pub fn write_path(&mut self, name: &str, path: PathBuf, file_type: &str) {
        let len = Self::get_header_len(name, path.file_name(), Some(file_type), &self.boundary);
        self.len += len;
        self.len += 2;

        let file_len = File::metadata(&File::open(&path).unwrap()).unwrap().len() as usize;
        self.len += file_len;

        self.fields
            .insert(name.to_string(), Field::File(path, file_type.to_string()));
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn boundary(&self) -> &str {
        &self.boundary
    }
    fn get_header_len(
        name: &str,
        filename: Option<&OsStr>,
        content_type: Option<&str>,
        boundary: &str,
    ) -> usize {
        let prefix = 2;
        let mut boundary_len = boundary.len();
        boundary_len += 2;
        boundary_len += prefix;
        //                             Content-Disposition: form-data; name="fuckyou"; filename="lib.rs"
        let name_len = format!("Content-Disposition: form-data; name=\"{}\"", name).len();
        let mut file_name_len = 0;

        if let Some(filename) = filename {
            file_name_len = format!("; filename=\"{}\"", filename.to_string_lossy()).len();
        }
        let back_return2 = 2;

        let mut content_tpye_len = 0;
        if let Some(content_type) = content_type {
            content_tpye_len = format!("Content-Type: {}\r\n", content_type).len();
        }
        let back_return = 2;
        boundary_len + name_len + file_name_len + back_return + back_return2 + content_tpye_len
    }
}
impl RequestBodyAdaptor for FormDataAdaptor {
    fn write(&mut self, writer: impl Write) {
        let mut form_data = FormData::new(writer, self.boundary.clone());
        for (name, field) in self.fields.iter() {
            match field {
                Field::File(path, file_type) => {
                    form_data.write_path(name, path, file_type).unwrap();
                }
                Field::Str(value) => {
                    form_data.write_field(name, value).unwrap();
                }
            }
        }
        form_data.finish().unwrap();
    }
    fn default_headers(&self) -> Vec<(String, String)> {
        let e = vec![
            ("Content-Length".to_string(), self.len().to_string()),
            (
                "Content-Type".to_string(),
                format!("multipart/form-data; boundary={}", self.boundary()),
            ),
        ];
        e
    }
}
