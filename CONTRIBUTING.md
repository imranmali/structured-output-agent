# Contributing

Thanks for helping make LLM output reliable. This guide covers how work flows
from an idea to a release.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,server,anthropic]'          # all extras, as CI installs
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

Run the same checks CI runs before pushing, in an environment with all
extras installed (`pip install -e '.[dev,server,anthropic]'`), so the HTTP
service and the Anthropic client are linted, type-checked and tested too:

```bash
ruff check .
ruff format --check .
mypy
pytest --cov                     # fails below 90% coverage (see pyproject.toml)
openspec validate --all --strict
```

Optional extra checks:

```bash
# Package: build, then run the CLI from the bare wheel (no extras)
python -m build && pip install dist/*.whl && structured-agent demo

# Dependency audit, as the CI `audit` job runs it
python -m venv /tmp/audit-env
/tmp/audit-env/bin/pip install -e '.[server,anthropic]'
/tmp/audit-env/bin/pip freeze --exclude-editable > /tmp/requirements-audit.txt
pip install pip-audit
pip-audit --strict --no-deps --disable-pip -r /tmp/requirements-audit.txt

# Container: the demo, then the HTTP service and its probes
docker build -t structured-output-agent .
docker run --rm structured-output-agent demo
docker run --rm -d --name soa -p 8000:8000 \
  -e SOA_MODELS=structured_output_agent.cli:WeatherReport structured-output-agent
curl -fsS localhost:8000/healthz && curl -fsS localhost:8000/v1/models
docker stop soa
```

## Pull requests

- Fill in the PR template, including which OpenSpec change and tasks the PR
  covers.
- Add an entry under `## [Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) for
  every user-visible change (`Added`, `Changed`, `Deprecated`, `Removed`,
  `Fixed`, `Security`), following [Keep a Changelog](https://keepachangelog.com/).
  Internal-only changes (CI tweaks, refactors, tests) need no entry.
- CI must pass:
  - `.github/workflows/ci.yml`: lint, format and mypy; tests on Python
    3.10-3.14 with the coverage gate; `pip-audit` on the runtime dependencies;
    OpenSpec validation; the package build with a CLI smoke test; and the
    Docker image build with CLI and HTTP service smoke tests and a Trivy scan
    (fails on fixable critical CVEs).
  - `.github/workflows/codeql.yml`: CodeQL for Python and for the workflows.
- Report security problems privately as described in
  [`SECURITY.md`](SECURITY.md), not in a PR or issue.

### Recommended branch protection for `main`

Configure under **Settings -> Branches** (or a ruleset) for `main`:

- Require a pull request before merging (at least one approval; require
  review from Code Owners, see `.github/CODEOWNERS`; dismiss stale approvals
  on new commits).
- Require status checks to pass, and require branches to be up to date
  before merging. Required checks:
  - `Lint, format and type-check`
  - `Tests (Python 3.10)` through `Tests (Python 3.14)`
  - `OpenSpec validation`
  - `Build distributions`
  - `Docker image`
  - `Dependency audit`
  - `CodeQL (python)` and `CodeQL (actions)`
- Require conversation resolution before merging.
- Block force pushes and deletion of `main`; do not allow bypassing the
  rules.

## Release process

Releases are cut from `main` by pushing a version tag. Versions follow
[Semantic Versioning](https://semver.org/).

1. On a `chore/*` branch, prepare the release and merge the PR:
   - Bump `__version__ = "X.Y.Z"` in `src/structured_output_agent/__init__.py`.
     It is the single source of the version: `pyproject.toml` reads it through
     hatch's dynamic versioning, so there is nothing to change there.
   - In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`,
     add a new empty `## [Unreleased]` above it, and update the link
     references at the bottom (`[Unreleased]` compares `vX.Y.Z...HEAD`;
     `[X.Y.Z]` compares the previous tag with `vX.Y.Z`).
2. Tag the merge commit on `main` and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

3. The CD workflow (`.github/workflows/cd.yml`) then:
   - verifies the tag matches `__version__` and that `CHANGELOG.md` has a
     `## [X.Y.Z]` section (the release job fails otherwise),
   - builds the sdist and wheel and publishes a GitHub Release with
     auto-generated notes and the distributions attached,
   - builds and pushes the multi-arch (`linux/amd64`, `linux/arm64`)
     container image to `ghcr.io/imranmali/structured-output-agent` tagged
     `X.Y.Z` and `sha-<commit>`, with SBOM and provenance attestations,
   - publishes the distributions to PyPI, if enabled (see below).

Every push to `main` also publishes the image tagged `main`, `latest` and
`sha-<commit>`.

### Publishing to PyPI (optional)

PyPI publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so no API token is stored in the repository. It is off until you enable it:

1. On PyPI, add a trusted publisher for the `structured-output-agent` project
   (use a "pending publisher" before the first upload) with owner
   `imranmali`, repository `structured-output-agent`, workflow `cd.yml` and
   environment `pypi`.
2. In the GitHub repository, create the environment `pypi` under
   **Settings -> Environments**. Optionally restrict it to `v*` tags and add
   required reviewers so every upload is approved by hand.
3. Set the repository variable `PUBLISH_TO_PYPI` to `true` under
   **Settings -> Secrets and variables -> Actions -> Variables**.

The `pypi` job in `cd.yml` then runs after the GitHub Release on each `v*`
tag and uploads the same sdist and wheel. Set the variable to anything other
than `true` (or delete it) to stop publishing.
