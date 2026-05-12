# Contributing

We welcome all contributions. You don't need to be an expert to help out.

## Community & Discussions

Have a question or idea? Head over to [GitHub Discussions](https://github.com/Jebel-Quant/linalg/discussions).

## Checklist

Contributions are made through [pull requests](https://docs.github.com/en/pull-requests).
Before sending a pull request:

- Run `make fmt` to lint and format your code
- Run `make test` and ensure all tests pass
- If you changed dependencies in `pyproject.toml`, run `uv lock` and commit the updated `uv.lock`

## Building from source

```bash
make install
```

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/).

| Type       | When to use                                     |
|------------|-------------------------------------------------|
| `feat`     | New feature or capability                       |
| `fix`      | Bug fix                                         |
| `docs`     | Documentation only                              |
| `refactor` | Code change that is neither a fix nor a feature |
| `test`     | Adding or updating tests                        |
| `ci`       | CI / build system changes                       |
| `chore`    | Maintenance tasks (deps, tooling, config)       |
| `perf`     | Performance improvement                         |

## Code style

```bash
make fmt
```
