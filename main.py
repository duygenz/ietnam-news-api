# main.py

import feedparser
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from typing import List

# --- KHỞI TẠO ỨNG DỤNG VÀ MÔ HÌNH ---

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Vietnam News API",
    description="API để lấy tin tức và tạo vector toàn văn bản từ các trang tin tức Việt Nam.",
    version="1.0.0",
)

# Danh sách các nguồn RSS feed
RSS_FEEDS = [
    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "https://vneconomy.vn/chung-khoan.rss",
    "https://vneconomy.vn/tai-chinh.rss",
    "https://vneconomy.vn/thi-truong.rss",
    "https://vneconomy.vn/nhip-cau-doanh-nghiep.rss",
    "https://vneconomy.vn/tin-moi.rss",
]

# Tải mô hình sentence transformer nhỏ gọn, hỗ trợ đa ngôn ngữ (bao gồm tiếng Việt)
# 'paraphrase-multilingual-MiniLM-L12-v2' là một lựa chọn tốt vì nó nhỏ và hiệu quả.
print("Đang tải mô hình AI... Vui lòng chờ.")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Tải mô hình thành công!")


# --- CÁC HÀM XỬ LÝ ---

def get_full_article_text(url: str) -> str:
    """
    Hàm này truy cập vào URL của bài báo và trích xuất toàn bộ nội dung văn bản.
    Lưu ý: Logic bóc tách sẽ khác nhau cho mỗi trang web.
    """
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Logic cho vneconomy.vn
        if "vneconomy.vn" in url:
            content_div = soup.find("div", class_="detail__content")
            if content_div:
                return content_div.get_text(separator="\n", strip=True)

        # Logic cho cafef.vn
        elif "cafef.vn" in url:
            content_div = soup.find("div", id="mainContent")
            if content_div:
                return content_div.get_text(separator="\n", strip=True)
                
        # Trả về chuỗi rỗng nếu không tìm thấy nội dung
        return ""

    except requests.RequestException as e:
        print(f"Lỗi khi truy cập {url}: {e}")
        return ""
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý {url}: {e}")
        return ""


# --- ĐỊNH NGHĨA API ENDPOINT ---

@app.get("/news", summary="Lấy danh sách tin tức kèm vector")
async def get_news_with_vectors():
    """
    Endpoint này lấy tin tức mới nhất từ các nguồn RSS đã định nghĩa,
    trích xuất nội dung đầy đủ và tạo vector cho từng bài viết.
    """
    all_articles = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                full_text = get_full_article_text(entry.link)
                
                # Chỉ xử lý những bài báo có nội dung
                if full_text:
                    # Tạo vector từ nội dung đầy đủ
                    vector = model.encode(full_text).tolist()
                    
                    all_articles.append({
                        "source": feed.feed.title,
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.get("published", "N/A"),
                        "summary": entry.summary,
                        "full_text_vector": vector,
                    })
        except Exception as e:
            print(f"Lỗi khi xử lý RSS feed {feed_url}: {e}")
            continue # Bỏ qua feed bị lỗi và tiếp tục với các feed khác

    if not all_articles:
        raise HTTPException(status_code=500, detail="Không thể lấy được tin tức từ bất kỳ nguồn nào.")
        
    return {"articles": all_articles}


# Lệnh để chạy cục bộ: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
