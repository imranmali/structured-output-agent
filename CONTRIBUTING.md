# Contributing

Thanks for helping make LLM output reliable. This guide covers how work flows
from an idea to a release.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pip install pre-commit && pre-commit install   # optional, runs ruff on commit
```

Python 3.10 or newer is required. The OpenSpec CLI needs Node.js 20.19+
(CI uses Node 22):

```bash
npm install -g @fission-ai/openspec@1.14.3
```

## Branching model

- `main` is protected. Nothing is pushed to it directly; every change lands
  through a pull request, and a PR can only be merged when all CI checks are
  green.
- Branch from an up-to-date `main` and name the branch after the kind of work:

  | Prefix      | Use for                                   | Example                     |
  | ----------- | ----------------------------------------- | --------------------------- |
  | `feature/*` | New behaviour                             | `feature/tool-validation`   |
  | `fix/*`     | Bug fixes                                 | `fix/fenced-json-extraction` |
  | `ci/*`      | Workflows, build and release tooling      | `ci/github-actions`         |
  | `docs/*`    | Documentation and OpenSpec proposals      | `docs/openspec-proposal`    |
  | `chore/*`   | Maintenance, dependency bumps, scaffolding | `chore/scaffold`            |

- Keep branches short-lived and focused on one change. Rebase on `main` (or
  merge it in) if your branch falls behind.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <imperative summary>

<body: what changed and why>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`,
`ci`, `chore`. Mark breaking changes with `!` after the type/scope or a
`BREAKING CHANGE:` footer.

Examples:

```
feat(tools): validate tool-call arguments against Pydantic models
fix(parsing): pick the first decodable object in brace-scan extraction
docs(openspec): propose add-structured-output-agent change
test(retry): cover backoff when attempts are exhausted
ci(cd): publish GHCR image on main and GitHub Release on tags
build(docker): add multi-stage Dockerfile
chore: bump ruff to 0.17
feat(agent)!: return RunResult instead of raw model output
```

## OpenSpec workflow

Behavioural changes are specified before they are built, using
[OpenSpec](https://github.com/Fission-AI/OpenSpec). Specs live in
`openspec/specs/`; in-flight changes live in `openspec/changes/<change-name>/`
(`proposal.md`, `design.md`, `tasks.md` and delta specs under `specs/`).

1. **Propose.** Create the change and its artifacts, either with the
   `/opsx:propose` command in Claude Code (`/opsx:explore` first if you want to
   think the idea through) or by writing the files by hand. Check it with:

   ```bash
   openspec list                       # active changes
   openspec status --change <change-name>
   openspec validate <change-name> --strict
   ```

   Open the proposal as its own PR (typically on a `docs/*` branch) so it can
   be reviewed before implementation starts.

2. **Apply.** Implement the tasks in `tasks.md` on `feature/*` / `fix/*`
   branches with `/opsx:apply`, ticking each task as it is completed. If the
   design changes during implementation, update the artifacts in the same PR
   (`/opsx:update`).

3. **Archive.** Once the change is merged and deployed, archive it so its delta
   specs are folded into `openspec/specs/`:

   ```bash
   openspec archive <change-name>      # or /opsx:archive
   ```

   Use `--skip-specs` for tooling- or docs-only changes that have no spec
   deltas.

CI runs `openspec validate --all --strict` on every PR.

## Local quality gate

Run the same checks CI runs before pushing:

```bash
ruff check .
ruff format --check .
mypy
pytest --cov                     # fails below 90% coverage (see pyproject.toml)
openspec validate --all --strict
```

Optional extra checks:

```bash
python -m build && pip install dist/*.whl && structured-agent demo
docker build -t structured-output-agent . && docker run --rm structured-output-agent demo
```

## Pull requests

- Fill in the PR template, including which OpenSpec change and tasks the PR
  covers.
- CI (`.github/workflows/ci.yml`) must pass: lint, format, mypy, tests on
  Python 3.10-3.13 with the coverage gate, OpenSpec validation, the package
  build with a CLI smoke test, and the Docker image build.

## Release process

Releases are cut from `main` by pushing a version tag.

1. On a `chore/*` branch, bump the version in **both** places and merge the PR:
   - `pyproject.toml` -> `[project] version = "X.Y.Z"`
   - `src/structured_output_agent/__init__.py` -> `__version__ = "X.Y.Z"`
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

3. The CD workflow (`.github/workflows/cd.yml`) then:
   - verifies the tag matches the version in `pyproject.toml` (the release job
     fails otherwise),
   - builds the sdist and wheel and publishes a GitHub Release with
     auto-generated notes and the distributions attached,
   - builds and pushes the container image to
     `ghcr.io/imranmali/structured-output-agent` tagged `X.Y.Z` and
     `sha-<commit>`.

Every push to `main` also publishes the image tagged `main`, `latest` and
`sha-<commit>`.

Versions follow [Semantic Versioning](https://semver.org/).
