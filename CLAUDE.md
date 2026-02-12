# Zeus - Weather Generator (Pure Rust)

## Build & Test

- Build all: `cargo build`
- Build one crate: `cargo build -p zeus-arma`
- Test all: `cargo test --all`
- Test one crate: `cargo test -p zeus-wavelet`
- Single test: `cargo test test_name -- --exact`

## Lint & Format

- Format: `cargo fmt --all`
- Check format: `cargo fmt --all -- --check`
- Clippy: `cargo clippy --all --all-targets -- -D warnings`

## Workspace Layout

```text
zeus/              # root binary crate
crates/
  arma/            # zeus-arma: ARMA model
  wavelet/         # zeus-wavelet: wavelet transforms
```

New crates go in `crates/` and are auto-included via `members = ["crates/*"]`.

## Conventions

- Use `Result<T, E>` for fallible ops; no `.unwrap()` in library code
- `thiserror` for library errors, `anyhow` for the binary
- Doc comments (`///`) on all public items
- Unit tests in the same file; integration tests in `tests/`
- Prefer `impl Trait` over `dyn Trait` for internal APIs
- Mathematical variable names are acceptable in algorithm code (e.g., `phi`, `theta`, `sigma`)

## Workflow

1. Make changes
2. `cargo fmt --all && cargo clippy --all --all-targets -- -D warnings`
3. `cargo test --all`
4. Commit
