use super::{conflicting_env, parse_threads};

fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |n| {
        pairs
            .iter()
            .find(|(k, _)| *k == n)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn conflicting_thread_variable_is_reported() {
    assert_eq!(
        conflicting_env(4, env(&[("RAYON_NUM_THREADS", "2")])),
        Some(("RAYON_NUM_THREADS".into(), "2".into()))
    );
    assert_eq!(
        conflicting_env(1, env(&[("OMP_NUM_THREADS", "8")])),
        Some(("OMP_NUM_THREADS".into(), "8".into()))
    );
}

#[test]
fn equal_or_unset_variables_are_fine() {
    assert_eq!(conflicting_env(4, env(&[("RAYON_NUM_THREADS", "4")])), None);
    assert_eq!(conflicting_env(4, env(&[])), None);
}

#[test]
fn threads_flag_parses_and_rejects_zero() {
    let a = |s: &str| s.split(' ').map(String::from).collect::<Vec<_>>();
    assert_eq!(parse_threads(&a("bin")), Ok(1));
    assert_eq!(parse_threads(&a("bin --threads 4")), Ok(4));
    assert!(parse_threads(&a("bin --threads 0")).is_err());
    assert!(parse_threads(&a("bin --threads")).is_err());
}
