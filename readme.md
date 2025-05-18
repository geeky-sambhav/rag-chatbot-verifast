# RAG News Chatbot - Backend

This is the backend service for the RAG (Retrieval-Augmented Generation) News Chatbot. It's designed to ingest news articles, process user queries by retrieving relevant context from these articles, and generate answers using Google's Gemini LLM, while maintaining conversational history.

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Tech Stack & Justifications](#tech-stack--justifications)
4.  [Project Structure](#project-structure)
5.  [End-to-End Flow](#end-to-end-flow)
    * [Data Ingestion & Preprocessing](#data-ingestion--preprocessing)
    * [Embedding & Storage](#embedding--storage)
    * [API Request Handling (/chat)](#api-request-handling-chat)
    * [Chat History Management](#chat-history-management)
6.  [Setup and Running Instructions](#setup-and-running-instructions)
    * [Prerequisites](#prerequisites)
    * [Environment Variables](#environment-variables)
    * [Local Development Setup](#local-development-setup)
    * [Running Ingestion & Embedding Scripts](#running-ingestion--embedding-scripts)
7.  [Deployment (AWS EC2 Overview)](#deployment-aws-ec2-overview)
8.  [API Endpoints](#api-endpoints)
9.  [Caching & Performance (Redis)](#caching--performance-redis)
10. [Potential Improvements](#potential-improvements)

## 1. Overview

The backend serves as the brain of the chatbot. It exposes RESTful APIs for the frontend to interact with. Key responsibilities include:
* Ingesting and processing news articles to build a knowledge base.
* Embedding news content and user queries for semantic understanding.
* Storing and retrieving embeddings from a vector database.
* Managing chat session history.
* Interacting with a Large Language Model (Google Gemini) to generate responses based on retrieved context and conversation history.

## 2. Features

* **RAG Pipeline:** Retrieves relevant news snippets to augment LLM responses.
* **Conversational Context:** Maintains chat history per session for coherent follow-up interactions.
* **News Ingestion:** Processes news articles from RSS feeds.
* **Vector Search:** Uses Qdrant Cloud for efficient similarity search.
* **Session Management:**
    * Unique session ID generation.
    * Chat history retrieval per session.
    * Session clearing functionality.
* **RESTful API:** Provides endpoints for chat, history, and session clearing.

## 3. Tech Stack & Justifications

* **Programming Language:** Python 3.10+
    * *Justification:* Rich ecosystem for AI/ML, web development, and data processing. Mature libraries for all required tasks.
* **Web Framework:** FastAPI
    * *Justification:* High-performance ASGI framework, built-in data validation with Pydantic, automatic OpenAPI documentation, and easy to learn. Excellent for building robust APIs.
* **News Ingestion:**
    * `feedparser`: To parse RSS feeds for article URLs.
    * `news-please`: To extract main content from news article web pages.
    * *Justification:* Standard and effective libraries for web content retrieval and parsing.
* **Embedding Model:** Jina Embeddings (e.g., `jina-clip-v2`  via HTTP API)
    * *Justification:* As per assignment suggestion, provides a free tier for generating text embeddings. Using the HTTP API directly for simplicity.
* **Vector Database:** Qdrant Cloud
    * *Justification:* Managed vector database service, simplifies setup and maintenance. Offers efficient similarity search capabilities required for RAG.
* **LLM API:** Google Gemini API 2.0 flash
    * *Justification:* As per assignment requirement, a powerful generative model for producing final answers.
* **Session Cache & History:** Redis (running in Docker on EC2)
    * *Justification:* High-performance in-memory data store, ideal for session management and caching. Docker deployment on EC2 provides control and simplifies networking for this setup.
* **Containerization:** Docker
    * *Justification:* Ensures consistent environments from development to deployment, simplifies dependency management, and makes deployment to cloud services like EC2 straightforward.
* **Deployment Platform (Example):** AWS EC2 for the FastAPI application and Dockerized Redis, with Qdrant Cloud and other services being SaaS.
    * *Justification:* EC2 provides flexibility and control for running containerized applications. ElastiCache could be an alternative for managed Redis in a more production-heavy setup.

## 4. Project Structure

.├── .env                    # Environment variables (API keys, service endpoints) - NOT COMMITTED├── .gitignore              # Specifies intentionally untracked files that Git should ignore├── Dockerfile              # Instructions to build the Docker image for the FastAPI app├── embed_and_store.py      # Script to generate embeddings and store them in Qdrant├── ingest_news.py          # Script to fetch and preprocess news articles├── main.py                 # FastAPI application: API endpoints, Gemini interaction, Redis logic├── news_chunks_rss_newsplease.json # Example output of ingest_news.py (processed news chunks)├── rag_pipeline.py         # Core RAG logic: query embedding, Qdrant search, prompt construction└── requirements.txt        # Python dependencies
## 5. End-to-End Flow

### Data Ingestion & Preprocessing

1.  **Source:** News articles are sourced via RSS feeds using `ingest_news.py`.
2.  **Extraction:** The `news-please` library is used to extract the main textual content and title from each article URL obtained from the RSS feed.
3.  **Cleaning:** Basic text cleaning (e.g., removing excessive whitespace) is performed.
4.  **Chunking:** The cleaned article text is broken down into smaller, semantically coherent chunks (e.g., by paragraph). This is important for providing focused context to the LLM and respecting context window limits.
5.  **Output:** The processed chunks (text, source URL, title, chunk ID) are saved, typically to a JSON file (e.g., `news_chunks_rss_newsplease.json`).

### Embedding & Storage

1.  **Script:** `embed_and_store.py` handles this phase.
2.  **Load Chunks:** The script reads the processed text chunks from the JSON file.
3.  **Generate Embeddings:** Each text chunk is sent to the Jina Embeddings API ( `jina-clip-v2`) via an HTTP request. Jina returns a numerical vector (embedding) for each chunk.
4.  **Store in Vector DB:**
    * A connection is established to the Qdrant Cloud instance.
    * A collection is created (if it doesn't exist) with the appropriate vector size (matching Jina's output dimension) and distance metric (e.g., Cosine).
    * Each chunk's embedding vector, along with its original text and metadata (source URL, title, chunk ID), is stored as a point in the Qdrant collection.

### API Request Handling (`/chat`)

This is handled by the `main.py` FastAPI application.

1.  **Receive Request:** The frontend sends a POST request to the `/chat` endpoint with the `user_message` and an optional `session_id`.
2.  **Session ID Management:**
    * If no `session_id` is provided, a new unique ID is generated.
3.  **Retrieve Chat History:** (See [Chat History Management](#chat-history-management) below). Past messages for the current `session_id` are fetched from Redis.
4.  **Retrieve Relevant Context (RAG):**
    * The `user_message` is passed to functions in `rag_pipeline.py`.
    * **Query Embedding:** The `user_message` is embedded using the same Jina Embeddings model used for the news chunks.
    * **Vector Search:** The resulting query embedding is used to search the Qdrant Cloud collection for the top-k most semantically similar news chunks.
    * The text content of these retrieved chunks forms the "context".
5.  **Construct Prompt:**
    * A comprehensive prompt is constructed using the original `user_message`, the retrieved `chat_history`, and the `retrieved_contexts` (news snippets). This prompt instructs the LLM on how to formulate its answer.
6.  **Call Gemini LLM API:**
    * The constructed prompt is sent to the Google Gemini API (e.g., `gemini-1.5-flash-latest` model).
    * The LLM generates a response based on the provided query, history, and context.
7.  **Store Updated History:** The current `user_message` and the `bot_response` from Gemini are saved to Redis for the current session.
8.  **Return Response:** The `bot_response` and `session_id` (and context count for debugging) are sent back to the frontend.

### Chat History Management

* **Storage:** Redis (running in Docker on EC2) is used as an in-memory store for chat history.
* **Data Structure:** For each session, a Redis List is used with a key like `chat_history:<session_id>`.
* **Content:** Each item in the list is a JSON string representing a message: `{"role": "user" or "assistant", "content": "message text"}`.
* **TTL (Time-To-Live):** A TTL (e.g., 1 hour) is set on each session's Redis key. This ensures inactive sessions are automatically cleared from memory. The TTL is refreshed with each new message in the session.
* **Endpoints:**
    * `/history/{session_id}`: Fetches the entire chat history for a given session.
    * `/clear_session/{session_id}`: Deletes the chat history for a given session.

## 6. Setup and Running Instructions

### Prerequisites

* Python 3.10+
* Docker & Docker Compose (for local Redis/Qdrant if not using cloud versions for dev)
* AWS CLI (configured for ECR push and EC2 deployment)
* Access to:
    * Jina AI API Key
    * Google Gemini API Key
    * Qdrant Cloud account (URL and API Key)
    * (For AWS Deployment) An AWS account with permissions for EC2, ECR, ElastiCache (or ability to run Redis on EC2), VPC.

### Environment Variables

Create a `.env` file in the root of the backend project directory. This file should **NOT** be committed to Git. Add it to your `.gitignore`.

```env
# .env example
JINA_API_KEY="your_jina_api_key_here"
GEMINI_API_KEY="your_google_gemini_api_key_here"

# For Qdrant Cloud (used by embed_and_store.py and main.py/rag_pipeline.py)
QDRANT_HOST="your_qdrant_cloud_cluster_url_hostname_only" # e.g., xxx.cloud.qdrant.io
QDRANT_API_KEY="your_qdrant_cloud_api_key"
QDRANT_PORT="6334" # Or your Qdrant Cloud gRPC port
QDRANT_COLLECTION_NAME="news_articles_jina_http_v1" # Or your chosen collection name
VECTOR_SIZE="768" # Or the dimension of your Jina embedding model

# For Redis (used by main.py)
# If running Redis in Docker locally for dev:
# REDIS_HOST="localhost" # Or "my-redis-container" if using Docker Compose/network
# REDIS_PORT="6379"
# REDIS_PASSWORD="" # Leave empty or omit if no password for local dev Redis

# If using ElastiCache or Dockerized Redis on EC2 for deployed env:
# These would be set as environment variables in the Docker run command on EC2
# REDIS_HOST="your_elasticache_endpoint_or_redis_container_name_on_docker_network"
# REDIS_PORT="6379"
# REDIS_PASSWORD="your_redis_password_if_set"
Local Development SetupClone the repository.Create and activate a Python virtual environment:python -m venv venv
source venv/bin/activate  # On Linux/macOS
# .\venv\Scripts\activate    # On Windows
Install dependencies:pip install -r requirements.txt
Set up local Qdrant (Optional - if not using Qdrant Cloud for dev):docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage_local:/qdrant/storage qdrant/qdrant
Update .env with QDRANT_HOST=localhost, QDRANT_PORT=6333 (for REST) or 6334 (for gRPC), and no QDRANT_API_KEY.Set up local Redis (Optional - if not using cloud Redis for dev):docker run -p 6379:6379 --name my-local-redis -d redis
Update .env with REDIS_HOST=localhost, REDIS_PORT=6379.Ensure your .env file is populated with API keys and correct local service endpoints.Run the FastAPI application:uvicorn main:app --reload
The API will be available at http://127.0.0.1:8000 and docs at http://127.0.0.1:8000/docs.Running Ingestion & Embedding ScriptsThese scripts are typically run once to populate your vector database, or periodically to update it.Run News Ingestion (ingest_news.py):Ensure ingest_news.py is configured with your desired RSS feed URL(s).From your activated virtual environment:python ingest_news.py
This will produce a JSON file (e.g., news_chunks_rss_newsplease.json) with the processed news chunks.Run Embedding and Storage (embed_and_store.py):Ensure your .env file (or environment) has the correct JINA_API_KEY and connection details for your target Qdrant instance (local or Qdrant Cloud: QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, VECTOR_SIZE).Make sure the INPUT_JSON_FILE variable in embed_and_store.py points to the output of ingest_news.py.From your activated virtual environment:python embed_and_store.py
This will embed the chunks and store them in Qdrant.7. Deployment (AWS EC2 Overview)The application is designed to be containerized with Docker for deployment. The example deployment uses:FastAPI Application: Docker container running on an Amazon EC2 instance.Redis: Docker container running on the same EC2 instance, connected to the FastAPI app via a Docker network.Qdrant: Qdrant Cloud (SaaS).Key Deployment Steps (Summarized):Containerize: Build the Docker image for the FastAPI app using the provided Dockerfile.Push to ECR: Push the image to Amazon Elastic Container Registry (ECR).EC2 Setup:Launch an EC2 instance (e.g., Amazon Linux 2023).Install Docker.Attach an IAM Role to the EC2 instance with permissions to pull from ECR (AmazonEC2ContainerRegistryReadOnly).Configure Security Groups:EC2 SG: Allow inbound HTTP/S (port 80/443) from the internet, SSH (port 22) from your IP. Allow outbound to ECR, Jina, Gemini, Qdrant Cloud, and Redis (if on a separate instance or ElastiCache).(If using ElastiCache, its SG would allow inbound from EC2 SG on port 6379).Run Redis Container on EC2:Create a Docker network (e.g., rag_network).Run the official Redis image as a container on this network, with a volume mount for data persistence.Run FastAPI Container on EC2:Pull the image from ECR.Run the container, mapping host port 80 to container port 8000.Inject all necessary environment variables (JINA_API_KEY, GEMINI_API_KEY, QDRANT_HOST, QDRANT_API_KEY, QDRANT_PORT, REDIS_HOST="name_of_redis_container_on_docker_network", REDIS_PORT="6379") into the container at runtime.Data Ingestion: Run embed_and_store.py (locally or on EC2) configured to point to your Qdrant Cloud instance.8. API EndpointsThe API is documented via Swagger UI, available at /docs when the application is running.POST /chat: Main endpoint for sending user messages and receiving bot responses.Request Body: {"user_message": "string", "session_id": "string (optional)"}Response Body: {"bot_response": "string", "session_id": "string", "retrieved_context_count": "integer", "retrieved_contexts_details": "array (optional for debug)"}GET /history/{session_id}: Retrieves chat history for a given session.POST /clear_session/{session_id}: Clears chat history for a given session.GET /: Root endpoint with a welcome message.9. Caching & Performance (Redis)Session History: Redis is used as an in-memory cache to store the history of messages for each user session. This allows the LLM to have conversational context.Data Structure: Each session's history is stored as a Redis List, with each message being a JSON string.TTL (Time-To-Live): A TTL (e.g., SESSION_TTL_SECONDS = 3600 for 1 hour) is set on the Redis key for each session. This ensures that data for inactive sessions automatically expires, managing memory usage effectively. The TTL is refreshed upon each new message in an active session.Cache Warming: For this application's session history, traditional "cache warming" (pre-loading data before it's requested) is not directly applicable as sessions are user-specific and dynamic. The "warmth" comes from Redis being an in-memory store, ensuring fast reads/writes once a session is active. If sessions were persisted long-term in a slower database (like the optional SQL store), a cache warming strategy might involve loading frequently accessed or recently active sessions into Redis on application startup, but this is beyond the current scope.10. Potential ImprovementsStreaming Responses: Implement Server-Sent Events (SSE) or WebSockets to stream the LLM's response token by token to the frontend for a more interactive user experience.Advanced RAG:More sophisticated chunking strategies (e.g., sentence-window, recursive).Re-ranking of retrieved documents before sending to LLM.Query expansion or transformation.Error Handling & Resilience: More robust error handling for external API calls and service unavailability.Asynchronous Operations: Ensure all I/O-bound operations (API calls, database queries) are fully asynchronous to maximize FastAPI's performance.Scalability: For higher loads, consider deploying the FastAPI app with multiple Uvicorn workers (e.g., using Gunicorn as a process manager for Uvicorn) and potentially using an Application Load Balancer with an EC2 Auto Scaling Group. Replace Dockerized Redis with Amazon ElastiCache for better scalability and management.Persistent User Accounts & History: Implement user authentication and store chat histories persistently (e.g., in a SQL database) linked to user accounts.Automated News Corpus Updates: Schedule ingest_news.py and embed_and_store.py to run periodically to keep the knowledge base up-to-date.Monitoring & Logging: Integrate more comprehensive logging and monitoring (e.g., with AWS CloudWatch or other