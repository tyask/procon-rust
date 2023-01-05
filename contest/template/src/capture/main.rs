mod cargo_capture;

use std::{env, fs::{File, self}, io::{BufReader, Write, BufWriter}, error::Error };
use clap::*;
use cargo_capture::CargoCapture;
use itertools::Itertools;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().collect_vec();
    if args.len() > 1 && args[1] == "capture" {
        // cargoから実行時はargs[1]をスキップ. もっといい方法ある?
        args = args.iter()
            .take(1)
            .chain(args.iter().skip(2))
            .cloned().collect_vec();
    }

    let m = Command::new("cargo-capture")
        .version("0.1.0")
        .arg(arg!(--module <DIR>).required(true))
        .arg(arg!(--target <FILE>).required(true))
        .arg(arg!(--bkup))
        .get_matches_from(args);

    let module_project = m.get_one::<String>("module").expect("required").to_owned();
    let target_file = m.get_one::<String>("target").expect("required").to_owned();
    let bkup = m.get_one::<bool>("bkup").cloned().unwrap_or(false);

    let cap = CargoCapture::new(&module_project);
    let captured = cap.capture(BufReader::new(File::open(&target_file)?))?;

    if bkup { fs::copy(&target_file, format!("{}.bk", target_file))?; }

    let mut w = BufWriter::new(File::create(&target_file)?);
    w.write(captured.as_bytes())?;

    Ok(())
}
