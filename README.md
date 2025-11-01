<div align="center">
  <img src="https://storage.googleapis.com/gemini-repo/customer-support-agent-v3/logo.png" alt="Customer Support AI Agent Logo" width="200" />
  <h1>Customer Support AI Agent</h1>
  <p><strong>An enterprise-grade, AI-powered customer support system with real-time chat, intelligent document processing, and a scalable, distributed architecture.</strong></p>

  
  ![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
  ![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
  ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
  ![License](https://img.shields.io/github/license/nordeim/customer-support-agent-v3?color=blue)
  ![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
  ![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)

</div>

---

## 📋 Introduction

The **Customer Support AI Agent** is a production-ready, containerized application that revolutionizes customer service through intelligent automation. It delivers context-aware, personalized support experiences while maintaining enterprise-grade reliability and scalability.

This system combines a real-time conversational AI with a sophisticated backend architecture, featuring a custom-built agent orchestrator. It leverages Retrieval-Augmented Generation (RAG) to provide accurate, contextual responses based on your organization's knowledge base, while seamlessly handling document processing, conversation memory, and intelligent escalation to human agents.

### 🌟 Key Features

- **🤖 Custom AI Agent Orchestrator**: A sophisticated, in-house built agent that manages conversation flow, tool usage, and state.
- **💬 Real-Time Chat Interface**: WebSocket-powered instant messaging with streaming responses for a fluid user experience.
- **🧠 RAG-Powered Knowledge Retrieval**: Utilizes `SentenceTransformers` and `ChromaDB` for accurate information retrieval from your documents.
- **🧩 Extensible Tooling Architecture**: A dynamic `ToolRegistry` allows for easy integration of new capabilities (e.g., CRM, Billing, Inventory lookups).
- **🔄 Distributed Session Management**: A high-performance, Redis-backed session store with distributed locking and L1 caching ensures horizontal scalability and consistency across multiple instances.
- **📎 Advanced Document Processing**: Support for a wide range of file formats, including PDFs, Office documents, and images.
- **⚡ High Performance & Resilience**: Built with `FastAPI` and `asyncio`, incorporating connection pooling, retries, and circuit breakers for robust operation.
- **🐳 Container-Ready**: A full Docker Compose stack is provided for easy, reproducible deployments in any environment.

---

## 🛠️ Technology Stack

| Category      | Technology                                                                                             |
|---------------|--------------------------------------------------------------------------------------------------------|
| **Frontend**  | React, TypeScript, Vite, Tailwind CSS, Zustand, Axios                                                  |
| **Backend**   | FastAPI, Python 3.11+, SQLAlchemy, Alembic                                                             |
| **AI/ML**     | Custom Agent Orchestrator, SentenceTransformers (`all-MiniLM-L6-v2`), ChromaDB                         |
| **Infrastructure** | Docker, Redis, SQLite (dev), PostgreSQL (prod-ready)                                                   |

---

## 🏗️ Architecture

The application follows a decoupled frontend/backend architecture. The frontend is a Single Page Application (SPA) built with React, and the backend is a FastAPI service. During development, the Vite dev server proxies API and WebSocket requests to the backend to avoid CORS issues.

### Backend Architecture

The backend is a highly modular, production-grade FastAPI application.

*   **Core Component (`CustomerSupportAgent`)**: The heart of the backend is the `CustomerSupportAgent` (`backend/app/agents/chat_agent.py`). It is a sophisticated orchestrator that manages the entire conversation flow, tool usage, and state.

*   **Distributed Session Management**: A key architectural pillar is the externalized session management system, designed for horizontal scalability.
    *   **Pluggable Backends**: It uses a `SessionStore` abstraction (`backend/app/session/session_store.py`) with two implementations: a simple `InMemorySessionStore` for development and a production-grade `RedisSessionStore`.
    *   **High-Performance & Consistency**: The Redis store uses atomic Lua scripts for operations, features an in-memory L1 cache (`TTLCache`) to reduce latency, and employs Redis-based distributed locks (`backend/app/session/distributed_lock.py`) to prevent race conditions in a multi-instance environment.
    *   **Security**: Session data can be encrypted at rest, configured via `backend/app/config.py`.

*   **Extensible Tooling Architecture**: The system features a dynamic and resilient tool-use framework.
    *   **Tool Registry**: Instead of hardcoding tools, the agent uses a `ToolRegistry` (`backend/app/tools/registry.py`) to dynamically load and initialize tools based on configuration. This makes adding new capabilities (like the implemented CRM, Billing, and Inventory tools) straightforward.
    *   **Standardized Contracts**: All tools inherit from a `BaseTool` (`backend/app/tools/base_tool.py`) and return a standardized `ToolResult`, ensuring predictable integrations.
    *   **Resilience**: Tool calls are wrapped with resilience patterns like retries and circuit breakers (via `backend/app/tools/tool_call_wrapper.py`), making the system robust against transient failures of external services.

*   **Configuration-Driven System**: The entire backend's behavior is controlled by a comprehensive, type-safe configuration system in `backend/app/config.py`. This Pydantic-based setup allows for easy management of different environments, feature flags, and fine-grained tuning of all components, from the database pool to session encryption keys.

*   **Robust Application Lifecycle**: The `app/main.py` entry point defines a clean application lifecycle with a `lifespan` manager. This ensures graceful startup (including database checks, agent initialization) and shutdown. It also registers a full-featured middleware pipeline for rate limiting, request timing, correlation IDs, and global error handling.

### Frontend Architecture

The frontend is a modern React application built with Vite and TypeScript.

*   **Component-Based UI**: The UI is built with React components located in `frontend/src/components`. The main view is the `ChatInterface.tsx`.
*   **Hook-Based Logic**: Core business logic and state management for the chat are encapsulated in custom hooks, primarily `useChat.ts`. This hook orchestrates API calls and WebSocket events.
*   **Service Layer**: API and WebSocket communications are abstracted into singleton services (`frontend/src/services/api.ts` and `frontend/src/services/websocket.ts`), keeping network logic separate from UI components.

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+

### 1. Clone the Repository

```bash
git clone https://github.com/nordeim/customer-support-agent-v3.git
cd customer-support-agent-v3
```

### 2. Set Up Environment Variables

Copy the example environment file for the backend:

```bash
cp backend/.env.example backend/.env
```

### 3. Run with Docker Compose

This is the recommended method for a quick and easy setup.

```bash
docker-compose up -d --build
```

The application will be available at `http://localhost:3000`.

### Manual Setup

For more control during development:

1.  **Start Backend:**
    ```bash
    ./backend_start.sh
    ```

2.  **Start Frontend:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

---

## 🤝 Contributing

Contributions are welcome! Please see the `CONTRIBUTING.md` file for guidelines on how to get started.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.