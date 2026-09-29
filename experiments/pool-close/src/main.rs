//! Does an exit-handler notification or a join mark the end of a Rayon worker?
//! A worker thread-local has a destructor that blocks until the host releases it.
//! Pinned: rayon 1.11.0 / rayon-core 1.13.0 (Cargo.lock).

use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

struct Blocker {
    started: Sender<()>,
    release: Receiver<()>,
}
impl Drop for Blocker {
    fn drop(&mut self) {
        self.started.send(()).unwrap();
        self.release.recv_timeout(Duration::from_secs(5)).unwrap();
    }
}
thread_local! { static TLS: std::cell::RefCell<Option<Blocker>> = const { std::cell::RefCell::new(None) }; }

const BOUND: Duration = Duration::from_secs(2);

fn main() {
    let (exit_tx, exit_rx) = channel::<()>();
    let (dtor_started_tx, dtor_started_rx) = channel::<()>();
    let (release_tx, release_rx) = channel::<()>();
    let release_rx = Arc::new(Mutex::new(Some(release_rx)));

    // Owned pool: keep every worker's JoinHandle.
    let handles: Arc<Mutex<Vec<JoinHandle<()>>>> = Arc::default();
    let h = handles.clone();
    let exit_tx = Mutex::new(exit_tx);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .exit_handler(move |_| exit_tx.lock().unwrap().send(()).unwrap())
        .spawn_handler(move |t| {
            h.lock().unwrap().push(std::thread::Builder::new().spawn(move || t.run())?);
            Ok(())
        })
        .build()
        .unwrap();

    let rx = release_rx.clone();
    let started = dtor_started_tx.clone();
    pool.install(move || {
        TLS.with(|c| *c.borrow_mut() = Some(Blocker { started, release: rx.lock().unwrap().take().unwrap() }))
    });
    // Self-join guard: a worker must not join its own pool.
    assert_eq!(pool.install(|| rayon::current_thread_index()), Some(0));
    assert_eq!(pool.current_thread_index(), None); // host thread: joining is allowed

    drop(pool); // initiates shutdown, returns immediately
    exit_rx.recv_timeout(BOUND).expect("exit handler");
    dtor_started_rx.recv_timeout(BOUND).expect("TLS destructor started");
    println!("exit handler signaled; TLS destructor is still running");

    // Join on a helper thread so we can observe that it does not finish early.
    let (joined_tx, joined_rx) = channel::<()>();
    let joiner = std::thread::spawn(move || {
        for j in handles.lock().unwrap().drain(..) {
            j.join().unwrap();
        }
        joined_tx.send(()).unwrap();
    });
    match joined_rx.recv_timeout(Duration::from_millis(200)) {
        Err(RecvTimeoutError::Timeout) => println!("join still waiting while TLS destructor blocks: OK"),
        other => panic!("join returned before TLS teardown: {other:?}"),
    }
    release_tx.send(()).unwrap();
    joined_rx.recv_timeout(BOUND).expect("join after release");
    joiner.join().unwrap();
    println!("join returned only after TLS destructor finished: OK");
}
