# AI Agent Briefing: Customer Support System

## 1. Project WHAT: Core Purpose

This project is an enterprise-grade, AI-powered customer support system designed as a backend API service. Its primary function is to provide a sophisticated, real-time conversational interface for handling customer queries. The system is capable of understanding natural language, accessing a knowledge base to provide accurate answers, maintaining conversation history, and escalating to human agents when necessary.

**Key Features:**
*   **Real-time Conversational AI**: Engages users through a WebSocket-based chat interface.
*   **Retrieval-Augmented Generation (RAG)**: Utilizes a vector database (ChromaDB) and embedding models (`sentence-transformers`) to pull relevant information from a knowledge base and generate informed responses.
*   **Persistent, Distributed Sessions**: Manages user conversations across multiple server instances using a Redis-backed session store, ensuring a seamless user experience.
*   **Intelligent Tool Use**: The agent architecture is designed to use a registry of tools, allowing for extensible functionality (e.g., accessing CRM, billing, or inventory systems).
*   **Advanced System Health & Monitoring**: Includes comprehensive logging, telemetry (OpenTelemetry, Prometheus), and graceful shutdown procedures for high reliability.

## 2. Project WHY: The Problem It Solves

The goal of this project is to build a scalable, reliable, and intelligent customer support automation platform. It addresses the need for:
*   **24/7 Availability**: Providing instant support to customers at any time.
*   **Consistency**: Ensuring that answers to common questions are consistent and accurate.
*   **Scalability**: Handling a large volume of concurrent user conversations without performance degradation, made possible by its distributed session architecture.
*   **Efficiency**: Automating routine queries to free up human support agents for more complex issues.
*   **Extensibility**: Creating a flexible platform where new capabilities (tools) can be easily integrated to handle a wider range of customer needs.

## 3. Project HOW: Architecture and Design

The system is built on a modern Python stack, centered around the FastAPI framework, and follows a clean, modular architecture.

### 3.1. Technology Stack (from `requirements.txt`)

*   **Core Framework**: `FastAPI` for the API, with `Uvicorn` and `Gunicorn` for serving.
*   **Data Persistence**:
    *   **Relational Database**: `SQLAlchemy` as the ORM with `Alembic` for database migrations. The presence of `customer_support.db` suggests SQLite for development.
    *   **Vector Database**: `ChromaDB` for storing text embeddings for the RAG tool.
    *   **Cache & Session Store**: `Redis` for distributed session management and general-purpose caching.
*   **AI/ML**:
    *   **LLM Integration**: The `openai` library indicates direct integration with OpenAI's models.
    *   **Embeddings**: `sentence-transformers`, `torch`, and `transformers` are used to create vector embeddings from text for the RAG system.
*   **System Resilience & Performance**:
    *   **Asynchronous Operations**: Built on Python's `asyncio`, using libraries like `aiohttp` and `aiofiles`.
    *   **Retry & Circuit Breakers**: `tenacity` and `aiobreaker` are included to build resilient connections to external services.
*   **Monitoring & Telemetry**:
    *   `OpenTelemetry`: For distributed tracing.
    *   `Prometheus`: For collecting and exposing metrics.
    *   `Sentry`: For error tracking and reporting.
*   **Security**: `cryptography`, `pyjwt`, and `passlib` for handling encryption, JWTs, and password hashing.

### 3.2. Application Core (`app/main.py`)

The `app/main.py` file is the heart of the application, orchestrating all components.

*   **Managed Lifecycle (`lifespan`)**: The application uses FastAPI's `asynccontextmanager` for `lifespan` to manage resources.
    *   **On Startup**: It performs a sequence of critical initializations: creates logging directories, initializes and verifies the database connection, connects to the Redis cache, sets up telemetry, and, most importantly, initializes the `CustomerSupportAgent`. It also launches a background task for periodic session cleanup. A series of startup health checks ensures all critical components are ready before the application starts accepting requests.
    *   **On Shutdown**: It handles `SIGTERM` and `SIGINT` signals to trigger a graceful shutdown, ensuring background tasks are completed and all connections (agent, cache, database) are closed cleanly.
*   **Middleware Pipeline**: Incoming requests are processed through a well-defined middleware stack (executed in reverse order of addition):
    1.  `CORSMiddleware`: Handles Cross-Origin Resource Sharing policies.
    2.  `RateLimitMiddleware`: Protects the API from excessive requests.
    3.  `TimingMiddleware`: Measures and logs request processing time.
    4.  `RequestIDMiddleware`: Injects a unique ID into every request for improved logging and tracing.
    5.  `ErrorHandlingMiddleware`: A global handler that catches all unhandled exceptions and returns a standardized JSON error response.

### 3.3. AI Agent & Tooling (`app/agents/` & `app/tools/`)

*   **Central Orchestrator**: The `CustomerSupportAgent` class (in `app/agents/chat_agent.py`) is the central intelligence of the system. It is initialized once at startup and is responsible for processing user messages.
*   **Tool-Using Architecture**: The agent is designed to use a collection of tools (e.g., `rag_tool`, `escalation_tool`). The code in `main.py` shows the agent loading the RAG tool (`agent.tools.get('rag')`). This design is highly extensible, allowing new capabilities to be added by simply creating new tool classes.
*   **Asynchronous Initialization**: The agent can initialize its components asynchronously (`agent.initialize_async()`), which is crucial for non-blocking startup when using a tool registry.

### 3.4. API and Routing (`app/api/`)

*   **Modular Routes**: The API is cleanly organized into modules:
    *   `health.router`: Exposes a `/health` endpoint for system health checks.
    *   `sessions.router`: Manages the lifecycle of chat sessions (create, get, etc.).
    *   `chat.router`: Handles the primary chat logic.
*   **Real-time Communication**: A WebSocket endpoint at `/ws` provides a persistent, bidirectional channel for real-time chat between the user and the AI agent.

### 3.5. Session Management (`app/session/`)

The session management system is designed for scalability and resilience.
*   **Distributed State**: By using Redis (`RedisSessionStore`), the application can maintain user session state across multiple, stateless server instances.
*   **Advanced Features**: The system is configured to support sophisticated session features, including:
    *   **Distributed Locking**: To prevent race conditions when modifying session data.
    *   **Encryption**: To secure sensitive session data at rest.
    *   **L1 Caching**: An in-memory cache layer for frequently accessed sessions to reduce Redis latency.
*   **Automatic Cleanup**: A background task runs periodically to find and remove expired sessions from the store, preventing data bloat.

### 3.6. Configuration (`app/config.py`)

*   **Centralized & Type-Safe**: The project uses Pydantic's `Settings` class to manage configuration. This provides a single, type-safe source of truth for all settings.
*   **Environment-Aware**: Settings are loaded from environment variables and `.env` files, allowing for easy configuration across different environments (development, testing, production) without code changes.
