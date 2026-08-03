//! Result collection, table formatting and CSV output.

use std::fmt::Write as _;
use std::fs;

/// One measurement.
#[derive(Clone, Debug)]
pub struct Row {
    pub case: String,
    pub group: String,
    pub dtype: String,
    pub engine: String,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub macs: u64,
    pub secs: f64,
    pub gflops: f64,
    /// Fraction of A's row blocks and B's column blocks that are regular.
    pub reg_a: f64,
    pub reg_b: f64,
    pub notes: String,
}

#[derive(Default)]
pub struct Results {
    pub rows: Vec<Row>,
}

impl Results {
    pub fn push(&mut self, r: Row) {
        self.rows.push(r);
    }

    pub fn get(&self, case: &str, dtype: &str, engine: &str) -> Option<&Row> {
        self.rows
            .iter()
            .find(|r| r.case == case && r.dtype == dtype && r.engine == engine)
    }

    pub fn write_csv(&self, path: &str) -> std::io::Result<()> {
        let mut s = String::from(
            "case,group,dtype,engine,m,n,k,macs,seconds,gflops,regular_a,regular_b,notes\n",
        );
        for r in &self.rows {
            let _ = writeln!(
                s,
                "{},{},{},{},{},{},{},{},{:.9},{:.4},{:.4},{:.4},{}",
                r.case,
                r.group,
                r.dtype,
                r.engine,
                r.m,
                r.n,
                r.k,
                r.macs,
                r.secs,
                r.gflops,
                r.reg_a,
                r.reg_b,
                r.notes
            );
        }
        fs::write(path, s)
    }
}

pub fn print_environment() {
    println!("host        : {}", hostname());
    println!(
        "target      : {} / {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    #[cfg(target_arch = "x86_64")]
    {
        let f = |n: &str, v: bool| if v { format!("{n} ") } else { String::new() };
        print!("cpu features: ");
        print!("{}", f("avx2", std::is_x86_feature_detected!("avx2")));
        print!("{}", f("fma", std::is_x86_feature_detected!("fma")));
        print!("{}", f("avx512f", std::is_x86_feature_detected!("avx512f")));
        print!(
            "{}",
            f("avx512dq", std::is_x86_feature_detected!("avx512dq"))
        );
        println!();
    }
    // The cache geometry the analytical blocking model reads, and which source
    // answered. A blocking parameter that came out of a probe should be
    // traceable to it, and a fallback to the built-in defaults — which would
    // make every derived parameter conservative — should be visible here rather
    // than inferred from a disappointing number.
    {
        use tensorcontract::kernel::cache;
        let h = cache::hierarchy();
        let one = |l: &cache::CacheLevel| {
            format!(
                "L{}={}KiB/{}-way/{}sh",
                l.level,
                l.size / 1024,
                l.ways,
                l.shared_by
            )
        };
        let levels: Vec<String> = [Some(h.l1d), h.l2, h.l3]
            .into_iter()
            .flatten()
            .map(|l| one(&l))
            .collect();
        println!(
            "caches      : {} (via {}), blocking={}",
            levels.join(" "),
            h.source.name(),
            cache::block_model().name()
        );
    }
    println!(
        "baselines   : tblis={} blas={}",
        cfg!(feature = "tblis"),
        cfg!(feature = "blas")
    );
    #[cfg(feature = "tblis")]
    println!(
        "tblis       : abi {}, {} thread(s)",
        crate::tblis::VERSION,
        unsafe { crate::tblis::tblis_get_num_threads() }
    );
}

fn hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

/// Right-aligned fixed-width table printer.
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(headers: &[&str]) -> Self {
        Table {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }
    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }
    pub fn print(&self) {
        let ncol = self.headers.len();
        let mut w: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for r in &self.rows {
            for (i, c) in r.iter().enumerate().take(ncol) {
                w[i] = w[i].max(c.len());
            }
        }
        let line: String = w
            .iter()
            .map(|&x| "-".repeat(x + 2))
            .collect::<Vec<_>>()
            .join("+");
        let fmt = |cells: &[String]| -> String {
            cells
                .iter()
                .enumerate()
                .take(ncol)
                .map(|(i, c)| format!(" {:>width$} ", c, width = w[i]))
                .collect::<Vec<_>>()
                .join("|")
        };
        println!("{}", fmt(&self.headers));
        println!("{line}");
        for r in &self.rows {
            println!("{}", fmt(r));
        }
    }
}
