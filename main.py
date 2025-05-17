import os
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import uuid
from google import genai
import redis # For Redis integration
import json # For serializing messages for Redis
from typing import List, Dict, Optional

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ChatMessage(BaseModel): # For storing in Redis and returning in history
    role: str # "user" or "assistant"
    content: str

class ChatQuery(BaseModel):
    user_message: str
    session_id: Optional[str] = None

class RetrievedContextDetail(BaseModel):
    source_title: Optional[str] = "N/A"
    source_url: Optional[str] = "N/A"
    text_snippet: str

class ChatResponse(BaseModel):
    bot_response: str
    session_id: str
    retrieved_context_count: int = 0
    retrieved_contexts_details: List[RetrievedContextDetail] = Field(default_factory=list)

class HistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None) # Add REDIS_PASSWORD to .env if needed
SESSION_TTL_SECONDS = 3600 # 1 hour for session expiry

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=False # Store bytes, decode when retrieving JSON
    )
    redis_client.ping() # Check connection
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.RedisError as e:
    logger.error(f"Could not connect to Redis: {e}. Chat history will not work.")
    redis_client = None

# --- Helper function to construct prompt with history ---
def construct_prompt_with_history(user_query: str, retrieved_contexts: list[str], chat_history: list[dict]) -> str:
    """
    Constructs a prompt for the LLM including chat history and retrieved contexts.
    Each item in chat_history is expected to be a dict: {"role": "user/assistant", "content": "message"}
    """
    history_str_parts = []
    for message in chat_history:
        role = "User" if message.get("role") == "user" else "Chatbot"
        history_str_parts.append(f"{role}: {message.get('content')}")
    
    history_for_prompt = "\n".join(history_str_parts)
    
    context_str = "\n\n---\n\n".join(retrieved_contexts)

    prompt = f"""You are a helpful news assistant. Please answer the user's question based on your knowledge and the provided news article snippets and conversation history.

Conversation History:
{history_for_prompt if history_for_prompt else "No previous conversation history."}

---
News Article Snippets for current query:
{context_str if retrieved_contexts else "No specific news snippets found for this query."}
---

Current User's Question: {user_query}

Answer:"""
    return prompt

# Import functions from your RAG pipeline script
try:
    from rag_pipeline import retrieve_relevant_chunks, construct_prompt_for_llm
except ImportError:
    logging.error("Failed to import from rag_pipeline.py. Make sure the file exists and is in the correct path.")
    # You might want to exit or raise a more critical error if rag_pipeline is essential at startup
    def retrieve_relevant_chunks(user_query: str, top_k: int = 3) -> list[str]:
        logging.error("rag_pipeline.retrieve_relevant_chunks is not available.")
        return []
    def construct_prompt_for_llm(user_query: str, retrieved_contexts: list[str]) -> str:
        logging.error("rag_pipeline.construct_prompt_for_llm is not available.")
        return f"User Query: {user_query}\nContext: [rag_pipeline not available]"

# --- Load Environment Variables ---
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG News Chatbot API",
    description="API for the RAG-powered chatbot using Jina, Qdrant, and Gemini.",
    version="0.1.0",
    # docs_url="/docs", # Default
    # redoc_url="/redoc" # Default
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# Allows your frontend (running on a different port/domain) to communicate with this API.
# Adjust origins as needed for production.
origins = [
    "http://localhost",         # Common for local development
    "http://localhost:3000",    # Default for Create React App
    "http://localhost:5173",    # Default for Vite React
    # Add your deployed frontend URL here if you deploy it
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Configure Google Gemini ---
# Note: Make sure to install google-genai package by running:
# pip install google-genai
# Or add it to your requirements.txt file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # Current model as of mid-2025

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. The application might not work correctly.")
else:
    try:
        # Try importing the Google GenAI SDK
        try:
            # Configure the client with API key
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Google Gemini client initialized successfully.")
        except ImportError:
            logger.error("Failed to import google.genai. Please install the package with: pip install google-genai")
    except Exception as e:
        logger.error(f"Error initializing Google Gemini client: {e}")

# --- API Endpoints ---
@app.get("/", tags=["General"])
async def read_root():
    """
    Root endpoint for the API.
    """
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the RAG News Chatbot API! Visit /docs for API documentation."}

@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def handle_chat_message(query: ChatQuery, request: Request): # Added Request for client IP logging
    """
    Handles a user's chat message, performs Retrieval-Augmented Generation (RAG),
    and gets a response from the Gemini LLM.
    """
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Received user message: '{query.user_message}' from {client_host} for session: {query.session_id}")

    if not gemini_client:
        logger.error("Gemini client is not initialized. Cannot process chat message.")
        raise HTTPException(status_code=503, detail="LLM service is unavailable.")
    
    current_session_id = query.session_id if query.session_id else str(uuid.uuid4())
    logger.info(f"Using session_id: {current_session_id}")

    # Get chat history from Redis
    chat_history_for_prompt: List[Dict[str, str]] = []
    if redis_client:
        try:
            # Get history using list operation
            history_key = f"chat_history:{current_session_id}"
            # Wrap the Redis call in a try block and handle all items individually
            try:
                # NOTE: There might be a linter warning here about Redis response not being iterable
                # This is likely due to type annotations in redis-py and how the linter sees them
                # The code should still work correctly at runtime despite the warning
                raw_history = redis_client.lrange(history_key, 0, -1)
                if raw_history:
                    for item_bytes in raw_history:
                        try:
                            if item_bytes:
                                message_dict = json.loads(item_bytes.decode('utf-8'))
                                chat_history_for_prompt.append(message_dict)
                        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as decode_err:
                            logger.error(f"Could not decode history item: {decode_err}")
            except Exception as redis_err:
                logger.error(f"Error accessing Redis history: {redis_err}")
                
            logger.info(f"Retrieved {len(chat_history_for_prompt)} messages from history for session {current_session_id}")
        except Exception as e:
            logger.error(f"Error retrieving chat history from Redis for session {current_session_id}: {e}")
            # Continue without history if Redis fails for retrieval

    # Retrieve relevant news chunks using RAG pipeline
    retrieved_contexts_for_llm: List[str] = []
    retrieved_contexts_details_for_response: List[RetrievedContextDetail] = []
    try:
        # Get relevant contexts
        temp_formatted_contexts = retrieve_relevant_chunks(query.user_message, top_k=3)
        retrieved_contexts_for_llm = temp_formatted_contexts
        
        if temp_formatted_contexts: # Attempt to parse for response details
            for ctx_str in temp_formatted_contexts:
                title, url = "N/A (Parse from context)", "N/A (Parse from context)"
                if "Source: " in ctx_str and "\nURL: " in ctx_str and "\nContent: " in ctx_str:
                    try:
                        title = ctx_str.split("Source: ")[1].split("\nURL: ")[0]
                        url = ctx_str.split("\nURL: ")[1].split("\nContent: ")[0]
                    except IndexError: pass
                retrieved_contexts_details_for_response.append(
                    RetrievedContextDetail(source_title=title, source_url=url, text_snippet=ctx_str[:150] + "...")
                )
    except Exception as e:
        logger.error(f"Error during RAG context retrieval: {e}")
    
    context_count = len(retrieved_contexts_for_llm)
    logger.info(f"Retrieved {context_count} relevant contexts for the current query.")
    if context_count > 0 and retrieved_contexts_for_llm: # Check if list is not empty
        logger.info("First retrieved context (snippet for LLM): " + retrieved_contexts_for_llm[0][:200] + "...")
    
    # Construct the prompt for Gemini with history
    prompt_for_gemini = construct_prompt_with_history(
        query.user_message, 
        retrieved_contexts_for_llm, 
        chat_history_for_prompt
    )
    logger.info(f"Prompt for Gemini (first 300 chars): {prompt_for_gemini[:300]}...")

    # Call Google Gemini API using chat functionality
    try:
        logger.info(f"Sending prompt to Gemini model: {GEMINI_MODEL_NAME}...")
        
        # Create a new chat for each request
        chat = gemini_client.chats.create(model=GEMINI_MODEL_NAME)
        
        # Send the message with context to Gemini
        response = chat.send_message(prompt_for_gemini)
        
        # Check if we got a response
        if response and response.text:
            bot_answer = response.text
            logger.info(f"Gemini response (first 200 chars): {bot_answer[:200]}...")
        else:
            logger.warning("Received empty response from Gemini")
            bot_answer = "I'm sorry, I couldn't generate a response at this time."
            
    except Exception as e:
        logger.error(f"Error calling Gemini API or processing its response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response from LLM: {str(e)}")

    # Store the conversation in Redis if available
    if redis_client:
        try:
            # Add user message to history
            user_message_json = json.dumps({"role": "user", "content": query.user_message})
            # Add bot message to history  
            bot_message_json = json.dumps({"role": "assistant", "content": bot_answer})
            
            # Add to Redis in correct order (newer messages at the front)
            history_key = f"chat_history:{current_session_id}"
            redis_client.lpush(history_key, bot_message_json)
            redis_client.lpush(history_key, user_message_json)
            
            # Set expiry on the chat history
            redis_client.expire(history_key, SESSION_TTL_SECONDS)
            
            logger.info(f"Stored conversation in Redis for session {current_session_id}")
        except Exception as e:
            logger.error(f"Error storing conversation in Redis: {e}")

    return ChatResponse(
        bot_response=bot_answer,
        session_id=current_session_id,
        retrieved_context_count=context_count,
        retrieved_contexts_details=retrieved_contexts_details_for_response
    )

retrieved_contexts_for_llm: List[str] = []
retrieved_contexts_details_for_response: List[RetrievedContextDetail] = []
try:
        # Assuming retrieve_relevant_chunks returns list of formatted strings as per previous setup
        temp_formatted_contexts = retrieve_relevant_chunks(query.user_message, top_k=3)
        retrieved_contexts_for_llm = temp_formatted_contexts
        
        if temp_formatted_contexts: # Attempt to parse for response details
            for ctx_str in temp_formatted_contexts:
                title, url = "N/A (Parse from context)", "N/A (Parse from context)"
                if "Source: " in ctx_str and "\nURL: " in ctx_str and "\nContent: " in ctx_str:
                    try:
                        title = ctx_str.split("Source: ")[1].split("\nURL: ")[0]
                        url = ctx_str.split("\nURL: ")[1].split("\nContent: ")[0]
                    except IndexError: pass
                retrieved_contexts_details_for_response.append(
                    RetrievedContextDetail(source_title=title, source_url=url, text_snippet=ctx_str[:150] + "...")
                )
except Exception as e:
        logger.error(f"Error during RAG context retrieval: {e}")
    
        context_count = len(retrieved_contexts_for_llm)
        logger.info(f"Retrieved {context_count} relevant contexts for the current query.")
        if context_count > 0 and retrieved_contexts_for_llm: # Check if list is not empty
          logger.info("First retrieved context (snippet for LLM): " + retrieved_contexts_for_llm[0][:200] + "...")


@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["Chat History"])
async def get_chat_history(session_id: str):
    logger.info(f"Fetching chat history for session_id: {session_id}")
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Chat history service is unavailable.")
    
    history_data: List[ChatMessage] = []
    try:
        history_items_bytes = redis_client.lrange(f"chat_history:{session_id}", 0, -1)
        if not history_items_bytes:
            logger.info(f"No history found for session_id: {session_id}")
            # Return empty history, not an error, if session doesn't exist or is empty
        for item_bytes in history_items_bytes:
            message_dict = json.loads(item_bytes.decode('utf-8'))
            history_data.append(ChatMessage(role=message_dict["role"], content=message_dict["content"]))
    except Exception as e:
        logger.error(f"Error retrieving history from Redis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve chat history.")
        
    return HistoryResponse(session_id=session_id, history=history_data)

@app.post("/clear_session/{session_id}", status_code=200, tags=["Chat History"])
async def clear_chat_session(session_id: str):
    logger.info(f"Attempting to clear chat history for session_id: {session_id}")
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Chat history service is unavailable.")
    
    try:
        deleted_count = redis_client.delete(f"chat_history:{session_id}")
        if deleted_count > 0:
            logger.info(f"Successfully cleared chat history for session_id: {session_id}")
            return {"message": f"Chat history for session {session_id} cleared successfully."}
        else:
            logger.info(f"No chat history found to clear for session_id: {session_id}")
            # Not an error if the session didn't exist
            return {"message": f"No chat history found for session {session_id}."}
    except Exception as e:
        logger.error(f"Error clearing history from Redis for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not clear chat history.")