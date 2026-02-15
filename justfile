# Default recipe
default: build

# Build everything
build: build-backend build-frontend

# Build backend only
build-backend:
    cd backend && cargo build --workspace

# Build frontend only
build-frontend:
    cd frontend && npm install && npm run build

# Start the server (backend serves frontend static files)
serve:
    cd backend && cargo run --bin server

# Run all tests
test: test-backend test-frontend

# Run backend tests only
test-backend:
    cd backend && cargo test --workspace

# Run frontend tests only
test-frontend:
    cd frontend && npm test

# Run e2e tests (Playwright)
test-e2e:
    cd frontend && npx playwright test

# Run reference test suite
test-reference:
    cd backend && cargo run --bin reference-tests

# Run benchmarks
bench:
    cd backend && cargo bench

# Watch mode (auto-rebuild backend on changes)
watch:
    cd backend && cargo watch -x 'run --bin server'
