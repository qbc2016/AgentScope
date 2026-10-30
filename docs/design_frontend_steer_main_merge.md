# Frontend Steer merge with main

## Agreed scope

Resolve the in-progress merge on `bc/frontend_steer`, retaining Steer while
adopting main's reference-backed queue storage, queue coordinator, reliability
fixes, realtime voice support and frontend response guards. Preserve unrelated
working files. Stage resolved files without committing or pushing.

## Checklist

- [x] Inspect the 25 conflicted files and agree on the merge strategy.
- [x] Resolve conflict markers and preserve both sides' features.
- [x] Adapt Steer writes and acknowledgement to reference-backed storage.
- [x] Preserve session and snapshot guards for asynchronous Steer responses.
- [x] Verify queue, steering, dispatcher, bus and session regression tests.
- [x] Verify frontend types, lint and production build (limits below).
- [x] Check the final diff and stage resolved files.

## Validation notes

The initial full frontend type check reports errors in the pre-existing,
untracked `src/components/chat/artifacts` directory: missing Monaco dependencies,
missing `./artifacts` and related types. These files are outside the merge scope.

- In the `agentscope` conda environment, 225 Python tests and 30 subtests
  passed. This includes queue storage/recovery, steering, both bus backends,
  dispatch/cancellation, session execution/events, auto-title generation,
  access policy, channel delivery, realtime service and team failure handling.
- Pytest plugin autoload was disabled because the environment's unrelated
  Phoenix plugin fails to import its FastMCP dependency.
- An additional `SessionNameOwnershipTest` HTTP test group blocked during
  application setup and was stopped. Its five tests were not validated;
  no production or test changes were made for that separate issue.
- TypeScript checked 201 root files successfully with the existing untracked
  artifacts directory excluded in memory; repository configuration was kept.
- Frontend ESLint reported no errors and two existing dependency warnings in
  `ChatViewport`. Vite production build passed, with its large-chunk warning;
  output was placed under the system temporary directory.
- Regression coverage checks that Steer keeps queue bodies in the payload
  registry, removes only acknowledged payloads, preserves other FIFO entries,
  and commits transitions even when live notification fails.
- All configured pre-commit hooks passed, including mypy, Black, flake8 and
  pylint. Conflict and whitespace checks passed. No commit or push was made.
