import json
import os
import requests # For making HTTP requests
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import time # For potential rate limiting delays

# --- Configuration ---
# Jina Configuration
JINA_MODEL_NAME = "jina-clip-v2"  # IMPORTANT: Choose a TEXT embedding model
                                                # e.g., "jina-embeddings-v2-base-en" or "jina-embeddings-v2-small-en"
                                                # "jina-clip-v2" is multimodal; if used, ensure it's what you need.
JINA_API_URL = "https://api.jina.ai/v1/embeddings"

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "news_articles_jina_http_v1" # New name to avoid conflict

# Determine vector size based on the chosen Jina model (CRITICAL)
if JINA_MODEL_NAME == "jina-clip-v2":
    VECTOR_SIZE = 1024
else:
    raise ValueError(f"VECTOR_SIZE for Jina model '{JINA_MODEL_NAME}' is not defined. Please add it.")

INPUT_JSON_FILE = "news_chunks_rss_newsplease.json" # From your successful Step 2

# --- Load Environment Variables (for Jina API Key) ---
load_dotenv()
JINA_API_KEY_FROM_ENV = os.getenv("JINA_API_KEY")
if not JINA_API_KEY_FROM_ENV:
    raise ValueError(
        "JINA_API_KEY not found in environment variables. "
        "Please set it in a .env file (e.g., JINA_API_KEY='your_key')."
    )

# --- Jina API Headers ---
jina_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API_KEY_FROM_ENV}" # Use the key loaded from .env
}

# --- Initialize Qdrant Client ---
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    print("Ensure Qdrant is running (e.g., via Docker).")
    exit()

def create_qdrant_collection_if_not_exists():
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Collection '{QDRANT_COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Collection '{QDRANT_COLLECTION_NAME}' does not exist. Creating it...")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )
        print(f"Collection '{QDRANT_COLLECTION_NAME}' created with vector size {VECTOR_SIZE}.")

# --- Function to get embeddings from Jina API ---
def get_jina_embeddings_http(texts_list, model_name):
    """
    Gets embeddings for a list of texts using Jina's HTTP API.
    Jina API can handle a list of inputs.
    """
    if not texts_list:
        return []

    # Prepare the input in the format Jina API expects for text
    # Your example showed `{"text": "..."}` for each text item.
    api_input = [{"text": text_content} for text_content in texts_list]

    data_payload = {
        "model": model_name,
        "input": api_input
    }

    print(f"Sending {len(texts_list)} texts to Jina API (model: {model_name})...")
    try:
        response = requests.post(JINA_API_URL, headers=jina_headers, data=json.dumps(data_payload), timeout=60) # Increased timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        response_json = response.json()

        if "data" not in response_json or not isinstance(response_json["data"], list):
            print(f"Error: Unexpected Jina API response format. 'data' field missing or not a list.")
            print(f"Response: {response_json}")
            return None # Or raise an error

        embeddings = [item["embedding"] for item in response_json["data"]]
        
        if len(embeddings) != len(texts_list):
            print(f"Error: Mismatch in number of embeddings ({len(embeddings)}) "
                  f"and input texts ({len(texts_list)}).")
            print(f"Response data: {response_json['data']}")
            return None # Or raise an error
            
        print(f"Successfully received {len(embeddings)} embeddings from Jina API.")
        return embeddings
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Jina API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Jina API Error Response: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Jina API Error Response (not JSON): {e.response.text}")
        return None # Or raise an error
    except Exception as e:
        print(f"An unexpected error occurred during Jina API call: {e}")
        return None


def load_embed_and_store_chunks():
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            news_chunks = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file '{INPUT_JSON_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{INPUT_JSON_FILE}'.")
        return

    print(f"Loaded {len(news_chunks)} chunks from '{INPUT_JSON_FILE}'.")

    # Process in batches to be mindful of API request sizes and potential rate limits
    # Jina's API can take a list of inputs, but very large lists might be problematic.
    # Let's define a batch size for how many texts we send in one API call.
    # The Jina docs or API limits page would specify optimal batch sizes. Let's assume 32-128 is reasonable.
    api_batch_size = 32 
    all_points_to_upsert = []
    
    chunks_to_process = [chunk for chunk in news_chunks if chunk.get('text')]
    if not chunks_to_process:
        print("No chunks with text content found to process.")
        return

    for i in range(0, len(chunks_to_process), api_batch_size):
        batch_chunks = chunks_to_process[i:i + api_batch_size]
        texts_in_batch = [chunk['text'] for chunk in batch_chunks]

        print(f"\nProcessing batch {i//api_batch_size + 1} of {len(chunks_to_process)//api_batch_size + 1} (size: {len(texts_in_batch)})")
        
        batch_embeddings = get_jina_embeddings_http(texts_in_batch, JINA_MODEL_NAME)

        if batch_embeddings is None or len(batch_embeddings) != len(texts_in_batch):
            print(f"Failed to get embeddings for batch {i//api_batch_size + 1}. Skipping this batch.")
            # You might want to add more robust error handling here, like retries or saving failed chunks.
            time.sleep(2) # Wait a bit before next batch if an error occurred
            continue 

        for j, chunk_data in enumerate(batch_chunks):
            payload = {
                "text": chunk_data['text'],
                "source_url": chunk_data['source_url'],
                "source_title": chunk_data['source_title'],
                "publish_date": chunk_data.get('publish_date'),
                "original_chunk_id": chunk_data['chunk_id']
            }
            
            # Determine a unique ID for Qdrant. Using the original news_chunks index.
            # This assumes the order in chunks_to_process maps back to news_chunks.
            # A more robust ID would be to hash chunk_data['chunk_id'] or use UUID.
            # For simplicity, if `i` is the start of the batch in `chunks_to_process`,
            # and `j` is the index within `batch_chunks`, we need a global unique ID.
            # Let's find the original index of chunk_data in news_chunks.
            original_news_chunks_index = -1
            for k_orig, orig_chunk in enumerate(news_chunks):
                if orig_chunk.get('chunk_id') == chunk_data.get('chunk_id'):
                    original_news_chunks_index = k_orig
                    break
            
            if original_news_chunks_index == -1:
                print(f"Could not find original index for chunk: {chunk_data.get('chunk_id')}. Skipping.")
                continue


            all_points_to_upsert.append(
                models.PointStruct(
                    id=original_news_chunks_index + 1, # Qdrant ID (1-based index from original list)
                    vector=batch_embeddings[j],
                    payload=payload
                )
            )
        # Optional: add a small delay between API batches if hitting rate limits
        if i + api_batch_size < len(chunks_to_process):
            time.sleep(1) # 1 second delay between Jina API calls for different batches

    if all_points_to_upsert:
        print(f"\nTotal points to upsert to Qdrant: {len(all_points_to_upsert)}")
        qdrant_batch_size = 100 # Batch size for upserting to Qdrant itself
        for i in range(0, len(all_points_to_upsert), qdrant_batch_size):
            batch_to_qdrant = all_points_to_upsert[i:i + qdrant_batch_size]
            try:
                qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=batch_to_qdrant,
                    wait=True
                )
                print(f"Upserted batch {i//qdrant_batch_size + 1} to Qdrant.")
            except Exception as e:
                print(f"Error upserting batch to Qdrant: {e}")
        print("All points processed for Qdrant upsertion.")
    else:
        print("No points were prepared for Qdrant upsertion.")


def verify_qdrant_collection():
    # (This function remains the same as before)
    try:
        collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"\n--- Qdrant Collection Info ---")
        print(f"Collection Name: {QDRANT_COLLECTION_NAME}")
        print(f"Status: {collection_info.status}")
        print(f"Points Count: {collection_info.points_count}")
        print(f"Vector Size Configured: {collection_info.config.params.vectors.size}")
        print(f"Distance Metric: {collection_info.config.params.vectors.distance}")
    except Exception as e:
        print(f"Error verifying Qdrant collection: {e}")

if __name__ == "__main__":
    print("Starting Step 3: Embedding Creation and Storage (Jina Embeddings via HTTP API)...")
    create_qdrant_collection_if_not_exists()
    load_embed_and_store_chunks()
    verify_qdrant_collection()
    print("\nStep 3 (Jina Embeddings via HTTP API) finished.")