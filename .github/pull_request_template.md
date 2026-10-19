## Summary

<!-- What does this PR do and why? One or two sentences. -->

## OpenSpec change

<!--
Link the change this PR implements, e.g. `openspec/changes/add-structured-output-agent`,
and the task numbers it completes (e.g. 2.1-2.4). Write "n/a" for CI, docs or chore
work that does not alter behaviour described by a spec.
-->

## Changes

-

## Testing

<!-- How was this verified? Paste the relevant commands and their results. -->

```bash
ruff check . && ruff format --check . && mypy
pytest --cov
openspec validate --all --strict
```

## Checklist

- [ ] Tests added or updated for new behaviour (coverage stays at or above 90%)
- [ ] Docs updated (README, docstrings, OpenSpec artifacts) where behaviour changed
- [ ] `openspec validate --all --strict` passes
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] Completed tasks are ticked in the change's `tasks.md`
