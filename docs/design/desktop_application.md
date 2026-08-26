# AgentScope Desktop Application Design

Status: Implemented

## Objective

Deliver a self-contained AgentScope desktop application for macOS, Windows,
and Linux. An installed application must not require a system Python runtime,
a separately installed AgentScope package, or Redis.

The implementation must preserve the existing browser-based Web UI workflow.

## Success criteria

- A packaged application starts its bundled backend and opens the UI without
  requiring Python or Redis on the host.
- Desktop data is persisted in the operating system's application data
  directory using SQLite, Qdrant Local, and the local filesystem.
- Only the Electron renderer that owns the per-launch secret can call the
  desktop backend.
- Production Electron windows keep Chromium web security enabled and cannot
  navigate to untrusted content.
- Backend startup, unexpected exit, and application shutdown behave correctly
  on macOS, Windows, and Linux.
- CI compiles, tests, packages, launches, probes, and stops the desktop backend
  on all three operating systems.
- The existing browser Web UI build still emits its normal web artifact.

## Architecture

```text
Electron main process
  |-- creates a random per-launch bearer token
  |-- resolves app.getPath("userData")
  |-- starts the bundled backend executable
  |-- enforces a single desktop application instance
  |-- owns backend startup and shutdown
  |
  +-- preload bridge (minimal configuration only)
        |
        +-- trusted renderer
              |
              +-- HTTP + bearer token --> 127.0.0.1:<dynamic port>

Bundled Python backend
  |-- AgentScope FastAPI application
  |-- AsyncSQLAlchemyStorage
  |-- SQLite database under Electron userData
  |-- LocalWorkspaceManager under Electron userData
  |-- CollectionPerKbManager backed by Qdrant Local
  |-- LocalBlobStore under Electron userData
  +-- InMemoryMessageBus
```

The backend binds an ephemeral loopback port itself and reports the selected
port to Electron through a machine-readable stdout handshake. This avoids the
race caused by selecting and releasing a free port before spawning Python.

## Security model

### Backend authentication

Electron generates a cryptographically random token for every launch and
passes it to the backend through the child process environment. The token is
never persisted.

A global desktop middleware requires `Authorization: Bearer <token>` and uses
a constant-time token comparison before a request reaches AgentScope. The
fixed `get_current_user_id` override is therefore reachable only after
authentication. Existing signed download and preview URLs remain usable
because those routes validate their scoped tokens instead.

The renderer receives the backend URL and bearer token through the isolated
preload bridge. This does not protect the token from a compromised trusted
renderer, so the Electron renderer must also be hardened.

### Renderer hardening

- Keep `contextIsolation` enabled.
- Keep `nodeIntegration` disabled.
- Keep `webSecurity` enabled in development and production.
- Install an Electron-only Content Security Policy.
- Deny new Electron windows by default.
- Send validated `http` and `https` external links to the system browser.
- Prevent top-level navigation away from the application document.
- Expose only immutable backend configuration through preload.

The browser Web UI must not receive the Electron-only Content Security Policy
or desktop credentials.

## Persistence

The desktop backend uses `AsyncSQLAlchemyStorage` with `aiosqlite`. Alembic
migrations run at startup so an existing desktop database can move forward
with the application.

Electron passes its platform-native user data directory to Python:

```text
<userData>/agentscope.db
<userData>/workspaces/
<userData>/qdrant/
<userData>/blobs/
```

This replaces the Redis dependency and avoids a hard-coded `~/.agentscope`
location that does not follow Windows and macOS application conventions.

Knowledge-base metadata is stored in SQLite, vectors are stored by Qdrant
Local, and uploaded source files are stored by `LocalBlobStore`. The desktop
build registers the default text parser only. PDF, Word, Excel, PowerPoint,
image, and audio parsing remain outside the desktop dependency set.

Qdrant Local is intended for small-scale local use. The desktop application
uses a single-instance lock so two backend processes cannot open the same
Qdrant data directory concurrently. Larger or shared deployments should run
the Web UI against a server-managed vector database instead.

## Backend packaging

PyInstaller builds an `onedir` backend named `agentscope-backend`. `onedir` is
preferred over `onefile` because it avoids extracting the full AgentScope
runtime on every application launch.

The build includes:

- AgentScope and its package data, including model cards and Alembic scripts;
- FastAPI and Uvicorn;
- SQLAlchemy, Alembic, and aiosqlite;
- Qdrant Client, its package metadata, and its local locking dependency;
- the platform-specific `rg` executable required by the built-in Grep tool;
- the local workspace tool runtime required by the desktop app.

Each operating system builds its own backend artifact. Electron Builder copies
the current platform's backend directory into `resources/backend`. Production
startup never falls back to a system Python interpreter. Development startup
uses an explicit Python executable or the active development environment.

## Process lifecycle

The Electron main process treats backend startup as a state machine:

1. Acquire the Electron single-instance lock.
2. Spawn the platform backend executable without a shell.
3. Parse the bounded stdout port handshake.
4. Probe `/health` with the bearer token.
5. Create the BrowserWindow only after readiness succeeds.
6. If the child exits before readiness, fail immediately and display the
   captured diagnostic output.
7. If the child exits after readiness, show a recoverable error and close or
   restart deliberately instead of leaving a broken UI.
8. On application quit, send a private shutdown command over the child stdin
   pipe so Uvicorn can complete its lifespan cleanup.
9. Wait for graceful exit and force termination of the process tree only
   after a bounded timeout.

Only one shutdown path owns the child process so `window-all-closed` and
`before-quit` cannot race each other.

## Web and Electron builds

Vite uses separate modes:

- the normal web build keeps the browser output directory and behavior;
- the Electron build uses a relative asset base, writes into the Electron
  staging directory, and injects the Electron-only CSP.

Package scripts use `pnpm exec` instead of `npx`, and the repository pins the
same pnpm major version used in CI. Node is upgraded to a version supported by
Electron 43.

## Testing and CI

### Python tests

- Assert the SQLite URL and platform data paths as complete values.
- Assert missing, malformed, and incorrect bearer tokens return 401.
- Assert a correct token returns the complete health response.
- Assert database schema creation/migration and workspace directory creation.
- Create a knowledge base, recreate the source application with the same data
  directory, and assert its SQLite metadata and Qdrant collection persist.

### Electron and frontend checks

- Type-check the Electron main and preload sources.
- Type-check and lint the frontend.
- Test handshake parsing and backend lifecycle helpers without launching a
  BrowserWindow where practical.
- Verify the Electron-specific frontend artifact contains its CSP while the
  web artifact remains unchanged.

### Packaging smoke test

A macOS, Windows, and Linux CI matrix will:

1. install the desktop build dependencies;
2. build the PyInstaller backend;
3. assert the bundled `rg` executable can start;
4. launch the backend with a temporary data directory and token;
5. read its port handshake and probe authenticated health;
6. create a knowledge base and gracefully stop the backend;
7. restart it with the same data directory and assert persistence;
8. gracefully stop it again;
9. run Electron Builder in unpacked-directory mode.

Installer signing and notarization use externally supplied CI credentials.
Unsigned pull-request smoke builds do not publish release artifacts.

## Implementation checklist

- [x] Refactor the Python entry point into a testable application factory.
- [x] Replace Redis with migrated SQLite storage under Electron `userData`.
- [x] Add persistent local knowledge bases with Qdrant Local and local blobs.
- [x] Prevent multiple desktop instances from sharing the Qdrant directory.
- [x] Add per-launch bearer-token validation.
- [x] Add a backend-owned dynamic-port stdout handshake.
- [x] Add PyInstaller build configuration and package-data collection.
- [x] Split Electron development and production backend resolution.
- [x] Replace shell command construction with executable-and-argument spawning.
- [x] Implement one cross-platform, awaited shutdown path.
- [x] Handle startup failure and unexpected backend exit visibly.
- [x] Enable Electron web security and add CSP/navigation/window restrictions.
- [x] Add bearer authentication to frontend desktop requests.
- [x] Separate browser and Electron Vite output modes.
- [x] Align Node and pnpm versions across local development and CI.
- [x] Add Python, Electron, and packaging smoke tests.
- [x] Verify knowledge-base persistence in source and CI smoke tests.
- [x] Add desktop CI for macOS, Windows, and Linux.
- [x] Replace the broken README link with committed desktop documentation.
- [x] Ignore generated Electron and PyInstaller artifacts.
- [x] Run formatting, lint, type checks, unit tests, and packaging verification.

## Non-goals

- Automatic application updates.
- Cloud synchronization of the local SQLite database.
- Multi-user authentication inside one desktop process.
- Bundling optional third-party model credentials or external sandbox engines.
- Shipping signing credentials in the repository.
