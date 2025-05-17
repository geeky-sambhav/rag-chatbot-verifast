import os
import requests # For Jina HTTP API
import json
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables (especially for JINA_API_KEY)
load_dotenv()

# --- Jina Configuration ---
JINA_API_KEY = os.getenv("JINA_API_KEY")
if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not found in environment variables.")

JINA_MODEL_NAME = "jina-clip-v2" # <<< MUST BE THE SAME MODEL USED IN STEP 3
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
jina_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    # --- Qdrant Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "news_articles_jina_http_v1" # <<< MUST BE THE SAME COLLECTION NAME USED IN STEP 3

    # Initialize Qdrant Client
try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"(RAG Pipeline) Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
        print(f"(RAG Pipeline) Error connecting to Qdrant: {e}")
    # In a real app, you might want to handle this more gracefully or retry
        raise e

def embed_query_with_jina(query_text: str) -> list[float] | None:
        """
        Embeds a single query string using Jina's HTTP API.
        """
        if not query_text:
            return None

        api_input = [{"text": query_text}] # API expects a list of inputs
        data_payload = {
            "model": JINA_MODEL_NAME,
            "input": api_input
        }

        print(f"Embedding query with Jina: '{query_text[:50]}...'")
        try:
            response = requests.post(JINA_API_URL, headers=jina_headers, data=json.dumps(data_payload), timeout=30)
            response.raise_for_status()
            response_json = response.json()

            if "data" in response_json and isinstance(response_json["data"], list) and len(response_json["data"]) > 0:
                embedding = response_json["data"][0].get("embedding")
                if embedding:
                    print("Query embedding received from Jina.")
                    return embedding
                else:
                    print(f"Error: 'embedding' field missing in Jina API response item for query.")
                    print(f"Response item: {response_json['data'][0]}")
                    return None
            else:
                print(f"Error: Unexpected Jina API response format for query embedding.")
                print(f"Response: {response_json}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error calling Jina API for query embedding: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during query embedding: {e}")
            return None
        

def retrieve_relevant_chunks(user_query: str, top_k: int = 3) -> list[str]:
        """
        Retrieves the text of the top_k most relevant chunks from Qdrant
        for a given user query.
        """
        if not user_query:
            return []

        # 1. Embed the user query
        query_embedding = embed_query_with_jina(user_query)

        if query_embedding is None:
            print("Failed to embed query. Cannot retrieve chunks.")
            return []

        # 2. Search Qdrant for relevant chunks
        try:
            print(f"Searching Qdrant collection '{QDRANT_COLLECTION_NAME}' for top {top_k} results.")
            search_results = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                # with_payload=True allows us to get the 'text' field
                # and other metadata we stored.
                with_payload=["text", "source_title", "source_url"] 
            )
            # print(f"Qdrant search results: {search_results}")
        except Exception as e:
            print(f"Error searching Qdrant: {e}")
            return []

        # 3. Extract the text from the search results
        retrieved_chunk_texts = []
        for hit in search_results:
            payload = hit.payload
            if payload and 'text' in payload:
                # Format context with source information
                context_item = f"Source: {payload.get('source_title', 'N/A')}\nURL: {payload.get('source_url', 'N/A')}\nContent: {payload['text']}"
                retrieved_chunk_texts.append(context_item)
                # Removed duplicate append of payload['text']
            else:
                print(f"Warning: Found a search hit without a text payload: ID {hit.id}")
        
        print(f"Retrieved {len(retrieved_chunk_texts)} relevant chunk texts.")
        return retrieved_chunk_texts

    # --- (Optional) Function to construct the prompt for Gemini ---
    # This will be used by your FastAPI endpoint later

def construct_prompt_for_llm(user_query: str, retrieved_contexts: list[str]) -> str:
        """
        Constructs a prompt for the LLM using the user query and retrieved contexts.
        """
        if not retrieved_contexts:
            # Fallback if no relevant context is found
            # You might want to handle this differently, e.g., by directly querying the LLM
            # or informing the user that no specific context was found.
            print("No relevant contexts found. Prompting LLM with query only.")
            # A simple prompt for this case:
            # context_section = "I could not find specific news articles in my current database related to your query."
            # prompt = f"""User Question: {user_query}
            #
            # Based on your general knowledge, please answer the user's question.
            #
            # Answer:"""
            # For now, let's try to always provide some context placeholder or ask it to rely on its knowledge.
            # For the assignment, it's better to show the RAG flow. If no context, Gemini will use its general knowledge.
            # We should always try to pass *something* as context, even if it's an empty string or a note.
            context_str = "\n\n---\n\n".join(retrieved_contexts)
            prompt = f"""Please answer the user's question based on the following news article snippets. If the snippets do not contain the answer, say that you couldn't find specific information in the provided articles and try to answer based on your general knowledge if appropriate.

News Article Snippets:
{context_str if retrieved_contexts else "No specific news snippets found for this query."}

User's Question: {user_query}

Answer:"""
       


        context_str = "\n\n---\n\n".join(retrieved_contexts) # Join contexts with a separator

        prompt = f"""Based on the following news article snippets, please answer the user's question.

News Article Snippets:
{context_str}

User's Question: {user_query}

Answer:"""
        return prompt

    # --- Example Usage (for testing this rag_pipeline.py script directly) ---
if __name__ == "__main__":
        print("\nTesting RAG Pipeline directly...")
        sample_query = "What are the latest developments in renewable energy?" # Change this to a query relevant to your news
        
        print(f"\nUser Query: {sample_query}")
        
        # 1. Retrieve relevant chunks
        retrieved_texts = retrieve_relevant_chunks(sample_query, top_k=3)
        
        if retrieved_texts:
            print("\n--- Retrieved Contexts ---")
            for i, text in enumerate(retrieved_texts):
                print(f"Context {i+1}: {text[:200]}...") # Print snippet of context
            
            # 2. Construct prompt for LLM (Gemini)
            final_prompt_for_gemini = construct_prompt_for_llm(sample_query, retrieved_texts)
            print("\n--- Prompt for LLM (Gemini) ---")
            print(final_prompt_for_gemini)
        else:
            print("\nNo relevant contexts retrieved for the sample query.")
            # Even if no context, you might still want to generate a prompt
            final_prompt_for_gemini = construct_prompt_for_llm(sample_query, [])
            print("\n--- Prompt for LLM (Gemini) - No Context ---")
            print(final_prompt_for_gemini)