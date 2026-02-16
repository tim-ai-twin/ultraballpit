//! HTTP + WebSocket Server
//!
//! This server provides the API endpoints and WebSocket connections for the
//! SPH fluid simulation frontend.

mod api;
mod runner;
mod state;
mod ws;

use axum::{
    routing::{get, post},
    Router,
};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use state::AppState;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting SPH simulation server");

    // Initialize kernel
    kernel::simulation::init();

    // Get port from environment or default to 3000
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    // Set up paths
    let configs_dir = PathBuf::from("configs");
    let geometries_dir = PathBuf::from("geometries");
    let frontend_dist = PathBuf::from("frontend/dist");

    // Create configs directory if it doesn't exist
    if !configs_dir.exists() {
        std::fs::create_dir_all(&configs_dir).expect("Failed to create configs directory");
    }

    // Create shared state
    let state = Arc::new(AppState::new(configs_dir, geometries_dir, port));

    // Configure CORS for development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build API router
    let api_router = Router::new()
        .route("/configs", get(api::list_configs))
        .route("/configs/:name", get(api::get_config))
        .route("/simulations", post(api::create_simulation))
        .route("/simulations/:id", get(api::get_simulation))
        .route("/simulations/:id/pause", post(api::pause_simulation))
        .route("/simulations/:id/resume", post(api::resume_simulation));

    // Build main application router
    let app = Router::new()
        .route("/health", get(health_handler))
        .nest("/api", api_router)
        .route("/ws/simulation/:id", get(ws::ws_simulation_handler))
        .with_state(state)
        .layer(ServiceBuilder::new().layer(cors));

    // Add static file serving if frontend dist exists
    let app = if frontend_dist.exists() {
        tracing::info!("Serving frontend from {:?}", frontend_dist);
        app.fallback_service(ServeDir::new(frontend_dist))
    } else {
        tracing::warn!("Frontend dist directory not found at {:?}", frontend_dist);
        app
    };

    // Run server
    let addr = format!("127.0.0.1:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap();

    tracing::info!("Server listening on {}", listener.local_addr().unwrap());

    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> &'static str {
    "OK"
}
