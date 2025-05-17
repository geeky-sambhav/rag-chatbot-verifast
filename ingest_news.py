import feedparser
from newsplease import NewsPlease
import json
import time

# --- Configuration ---
RSS_FEED_URL = "https://indianexpress.com/feed/"  # <<< IMPORTANT: REPLACE THIS!
# Example: RSS_FEED_URL = "https://indianexpress.com/feed/" # Verify this URL
MAX_ARTICLES_FROM_RSS = 60 # Get a few more URLs than target, as some might fail in news-please
MAX_ARTICLES_TO_PROCESS = 50 # Your target for successfully processed articles
OUTPUT_FILENAME = "news_chunks_rss_newsplease.json"

# --- 1. Get Article URLs from RSS Feed ---
def get_article_urls_from_rss(rss_url, max_urls):
    print(f"Fetching article URLs from RSS feed: {rss_url}")
    if not rss_url or rss_url == "YOUR_CHOSEN_RSS_FEED_URL_HERE":
        print("ERROR: RSS_FEED_URL is not set. Please update it in the script.")
        return []
        
    feed = feedparser.parse(rss_url)
    article_urls = []
    
    if feed.bozo:
        print(f"Warning: Feed may be malformed or inaccessible. Bozo exception: {feed.bozo_exception}")
        # If the feed URL itself is bad, you might get no entries.
        if not feed.entries:
            print(f"ERROR: Could not fetch or parse RSS feed. No entries found. Check the URL: {rss_url}")
            return []

    for entry in feed.entries:
        if len(article_urls) < max_urls:
            if hasattr(entry, 'link') and entry.link:
                article_urls.append(entry.link)
            else:
                print(f"Warning: RSS entry found without a link: {entry.get('title', 'No Title')}")
        else:
            break
    print(f"Found {len(article_urls)} article URLs from the RSS feed.")
    return article_urls

# --- 2. Fetch and Extract Content using NewsPlease ---
def fetch_content_with_newsplease(urls_to_fetch, limit_processed_articles):
    print(f"Attempting to process {len(urls_to_fetch)} URLs using NewsPlease (target: {limit_processed_articles} successful)...")
    processed_articles_data = []
    
    for i, url in enumerate(urls_to_fetch):
        if len(processed_articles_data) >= limit_processed_articles:
            print(f"Reached target of {limit_processed_articles} successfully processed articles.")
            break
        try:
            print(f"Processing URL ({i+1}/{len(urls_to_fetch)}): {url}")
            article = NewsPlease.from_url(url) # Timeout for news-please fetch
            
            if article and article.maintext and article.title:
                processed_articles_data.append({
                    "source_url": article.url or url,
                    "source_title": article.title,
                    "text": article.maintext,
                    "publish_date": str(article.date_publish) if article.date_publish else None,
                })
                print(f"Successfully extracted: {article.title}")
            else:
                error_message = "Maintext or title not found by NewsPlease."
                if not article: error_message = "NewsPlease returned None (download/parse error)."
                elif not article.maintext: error_message = "Maintext is empty."
                elif not article.title: error_message = "Title is empty."
                print(f"Warning: Could not adequately process {url}. {error_message}")
        except Exception as e:
            print(f"ERROR processing {url} with NewsPlease: {e}")
        
        # Be respectful: add a small delay between requests
        if i < len(urls_to_fetch) - 1:
            time.sleep(0.75) # Delay of 0.75 seconds

    return processed_articles_data

# --- 3. Clean Text (Minimal, as news-please often does a good job) ---
def clean_extracted_text(text):
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    cleaned_text = "\n\n".join([line for line in lines if line]) # Ensure paragraphs are separated by double newlines
    return cleaned_text.strip()

# --- 4. Chunk Text ---
def chunk_article_text(text, min_chunk_chars=100, max_chunk_chars=1500):
    raw_paragraphs = text.split("\n\n") # Assumes double newline paragraph separation
    chunks = []
    current_chunk = ""
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= max_chunk_chars:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk and len(current_chunk) >= min_chunk_chars:
                chunks.append(current_chunk)
            current_chunk = para 
            if len(current_chunk) > max_chunk_chars: # Handle oversized single paragraphs
                 if len(current_chunk) >= min_chunk_chars: # Add if it meets min length even if too long
                     chunks.append(current_chunk[:max_chunk_chars]) # Truncate
                 current_chunk = "" # Reset as it was too long and handled/truncated

    if current_chunk and len(current_chunk) >= min_chunk_chars:
        chunks.append(current_chunk)
    return chunks

# --- Main Ingestion Pipeline ---
def run_ingestion_pipeline():
    # Step 1: Get URLs from RSS Feed
    article_urls = get_article_urls_from_rss(RSS_FEED_URL, MAX_ARTICLES_FROM_RSS)

    if not article_urls:
        print("No article URLs fetched from RSS. Exiting ingestion.")
        return

    # Step 2: Fetch content for these URLs using NewsPlease
    articles_content_data = fetch_content_with_newsplease(article_urls, MAX_ARTICLES_TO_PROCESS)

    if not articles_content_data:
        print("No articles successfully processed by NewsPlease. Exiting.")
        return
        
    all_final_chunks = []
    articles_fully_chunked = 0

    # Step 3 & 4: Clean and Chunk text for each processed article
    for article_data in articles_content_data:
        cleaned_text = clean_extracted_text(article_data["text"])
        
        if not cleaned_text:
            print(f"Skipping article (empty after cleaning): {article_data['source_title']}")
            continue
            
        text_chunks = chunk_article_text(cleaned_text)

        if not text_chunks:
            print(f"Skipping article (no valid chunks): {article_data['source_title']}")
            continue

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_id = f"{article_data['source_url']}#chunk{chunk_index+1}"
            all_final_chunks.append({
                "source_url": article_data["source_url"],
                "source_title": article_data["source_title"],
                "publish_date": article_data.get("publish_date"),
                "chunk_id": chunk_id,
                "text": chunk_text
            })
        articles_fully_chunked += 1
    
    print(f"\n--- Ingestion Summary (RSS + NewsPlease) ---")
    print(f"Attempted to fetch {len(article_urls)} URLs from RSS (limit was {MAX_ARTICLES_FROM_RSS}).")
    print(f"NewsPlease attempted to process up to {min(len(article_urls), MAX_ARTICLES_TO_PROCESS)} of these URLs.")
    print(f"Successfully extracted content by NewsPlease for: {len(articles_content_data)} articles.")
    print(f"Articles successfully cleaned and chunked: {articles_fully_chunked}")
    print(f"Total text chunks created: {len(all_final_chunks)}")

    if all_final_chunks:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(all_final_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_final_chunks)} chunks to {OUTPUT_FILENAME}")
    else:
        print("No chunks were created. Please check your RSS feed and NewsPlease processing steps.")

if __name__ == "__main__":
    run_ingestion_pipeline()