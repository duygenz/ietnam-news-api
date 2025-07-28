# main.py

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import feedparser
import requests
from bs4 import BeautifulSoup
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Vietnam News API",
    description="An API to fetch news from Vietnamese RSS feeds and generate vector embeddings.",
    version="1.0.0"
)

# Load a small, efficient sentence-transformer model
# This model is multilingual and good for general purpose embeddings.
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# List of RSS feeds
RSS_FEEDS = [
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://vneconomy.vn/chung-khoan.rss",
    "https://vneconomy.vn/tai-chinh.rss",
    "https://vneconomy.vn/thi-truong.rss",
    "https://vneconomy.vn/nhip-cau-doanh-nghiep.rss",
    "https://vneconomy.vn/tin-moi.rss"
]

def get_full_article_text(url: str) -> str:
    """
    Fetches and extracts the main text content from a news article URL.
    This is a basic implementation and might need adjustments for specific sites.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the main content body. This often requires site-specific selectors.
        # Common selectors could be 'article', '.content', '#main-content', etc.
        # For vneconomy.vn, 'div.detail__content' seems to work well.
        # For cafef.vn, 'div.content' might be a good starting point.
        content_selectors = ['div.detail__content', 'div.content', 'article', 'div.main-content']
        article_body = None
        for selector in content_selectors:
            article_body = soup.select_one(selector)
            if article_body:
                break
        
        if article_body:
            # Remove script and style tags
            for script_or_style in article_body(['script', 'style']):
                script_or_style.decompose()
            # Get text and clean it up
            text = article_body.get_text(separator='\n', strip=True)
            return text
        return ""
    except requests.RequestException as e:
        print(f"Error fetching article at {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error parsing article at {url}: {e}")
        return ""

@app.get("/news", tags=["News"])
async def get_news_with_vectors():
    """
    Fetches the latest news from the predefined RSS feeds,
    extracts the full article content, and generates vector embeddings.
    """
    all_news = []
    for feed_url in RSS_FEEDS:
        parsed_feed = feedparser.parse(feed_url)
        for entry in parsed_feed.entries[:5]:  # Limit to 5 entries per feed for speed
            full_text = get_full_article_text(entry.link)
            
            if full_text:
                # Generate vector embedding for the full article context
                vector = model.encode(full_text).tolist()
                
                news_item = {
                    "source": parsed_feed.feed.title,
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", "N/A"),
                    "summary": entry.summary,
                    "full_text_vector": vector
                }
                all_news.append(news_item)
                
    return {"news": all_news}

if __name__ == "__main__":
    # To run this locally: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

