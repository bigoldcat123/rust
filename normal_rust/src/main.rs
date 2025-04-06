use encoding::{Encoding, all::GBK};
use std::{
    ffi::OsStr, fs::{self, File}, io::{self, Read, Write}, path::Path, process::Command
};
fn main() {
    let os = OsStr::new("hell");
    let e = format!("{}",os.to_str().unwrap());

}
fn do_dir<P: AsRef<Path>>(dir_path: P) -> Result<(), Box<dyn std::error::Error>> {
    let dir = fs::read_dir(dir_path)?;
    for file_path in dir {
        let srt_path = file_path?.path();
        if let Some(ext) = srt_path.extension() {
            if ext == "srt" {
                let mut mkv_path = srt_path.clone();
                let srt_file_name = srt_path.file_name().unwrap().to_string_lossy();
                let mkv_file_name = srt_file_name.replace(".srt", ".mkv");
                mkv_path.pop();
                mkv_path.push(mkv_file_name);
                ffmpeg(mkv_path.to_str().unwrap(), srt_path.to_str().unwrap());
            }
        }
    }
    Ok(())
}
fn ffmpeg(mvk: &str, srt: &str) {
    let e = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(mvk)
        .arg("-i")
        .arg(srt)
        .args([
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-map",
            "1:0",
            "-c",
            "copy",
            "-metadata:s:s:0",
            "language=chi",
            "-metadata:s:s:0",
            "title=\"简体中文\"",
            &mvk.replace("不死者之王", "overload"),
        ])
        .output()
        .unwrap();
    // io::stdout().write_all(&e.stdout).unwrap();
    io::stdout().write_all(&e.stderr).unwrap();
}

fn re_formate(path: &str) {
    let mut res = vec![];
    let mut e = File::open(path).unwrap();
    e.read_to_end(&mut res).unwrap();
    let e = GBK.decode(&res, encoding::DecoderTrap::Strict).unwrap();
    let mut file = File::create(path).unwrap();
    file.write_all(e.as_bytes()).unwrap();
}
