"""
Daily News Fetching Script
Runs automatically via GitHub Actions
Fetches news, processes with AI, saves to Google Sheets
"""

import os
import json
import time
import requests
import feedparser
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai

# ============================================
# 📋 CONFIGURATION
# ============================================

# Get credentials from environment variables
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
SHEET_ID = os.environ.get('SHEET_ID')
NEWSDATA_API_KEY = os.environ.get('NEWSDATA_API_KEY')
GOOGLE_CREDENTIALS_JSON = os.environ.get('GOOGLE_CREDENTIALS_JSON')

# RSS Feeds (backup)
RSS_FEEDS = {
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "PIB": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1"
}

# ============================================
# 🤖 AI CONFIGURATION (Groq)
# ============================================

def init_groq():
    """Initialize Groq client"""
    return openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )

groq_client = init_groq()

def get_groq_response(system_prompt, user_content, json_mode=False):
    """Call Groq Llama-3"""
    try:
        if json_mode:
            system_prompt += " RETURN ONLY VALID JSON. NO MARKDOWN. NO CODE BLOCKS."
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return None

# ============================================
# 📰 NEWS FETCHING
# ============================================

def fetch_newsdata_articles():
    """Fetch from NewsData.io"""
    if not NEWSDATA_API_KEY:
        print("⚠️ NewsData.io key not found, skipping...")
        return []
    
    try:
        url = "https://newsdata.io/api/1/news"
        params = {
            'apikey': NEWSDATA_API_KEY,
            'country': 'in',
            'language': 'en',
            'category': 'politics,top,world',
            'size': 10
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success' and 'results' in data:
                articles = []
                for article in data.get('results', []):
                    description = article.get('description') or article.get('content') or ''
                    content = description[:800] if description else 'No content'
                    
                    articles.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'title': article.get('title', 'No title'),
                        'source': (article.get('source_id') or 'Unknown').title(),
                        'link': article.get('link', ''),
                        'content': content
                    })
                print(f"✅ Fetched {len(articles)} articles from NewsData.io")
                return articles
        
        print(f"⚠️ NewsData.io returned status {response.status_code}")
        return []
    except Exception as e:
        print(f"❌ NewsData.io error: {e}")
        return []

def fetch_rss_feeds():
    """Fetch from RSS feeds"""
    articles = []
    
    for source, url in RSS_FEEDS.items():
        try:
            print(f"📡 Fetching from {source}...")
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:5]:
                articles.append({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'title': entry.get('title', 'No title'),
                    'source': source,
                    'link': entry.get('link', ''),
                    'content': entry.get('summary', entry.get('description', ''))[:800]
                })
            
            print(f"✅ Got {len(feed.entries[:5])} articles from {source}")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed to fetch from {source}: {e}")
    
    return articles

# ============================================
# 🤖 AI PROCESSING
# ============================================

def analyze_relevance(title, content):
    """Score article relevance 1-10"""
    prompt = f"Title: {title}\nContent: {content[:500]}"
    system = """Rate UPSC/SSC relevance 1-10.
    Return JSON: {"score": number}"""
    
    response = get_groq_response(system, prompt, json_mode=True)
    
    try:
        cleaned = response.replace("```json", "").replace("```", "").strip()
        score = json.loads(cleaned)['score']
        return min(max(int(score), 1), 10)
    except:
        return 5

def generate_summary(title, content):
    """Generate exam-focused summary"""
    prompt = f"Title: {title}\nContent: {content[:1000]}"
    system = """UPSC tutor. Write 3-4 line summary with key facts and syllabus area."""
    
    return get_groq_response(system, prompt) or "Summary failed"

def generate_mcq(title, content):
    """Generate MCQ in JSON format"""
    prompt = f"Title: {title}\nContent: {content[:1000]}"
    system = """Create UPSC MCQ. Return JSON:
    {
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "correct": "A",
        "explanation": "..."
    }"""
    
    response = get_groq_response(system, prompt, json_mode=True)
    
    try:
        cleaned = response.replace("```json", "").replace("```", "").strip()
        json.loads(cleaned)  # Validate
        return cleaned
    except:
        return json.dumps({
            "question": "Failed",
            "options": {"A": "N/A", "B": "N/A", "C": "N/A", "D": "N/A"},
            "correct": "A",
            "explanation": "Generation failed"
        })

def process_articles(articles):
    """Process articles with AI"""
    processed = []
    
    print(f"\n🤖 Processing {len(articles)} articles with Groq AI...")
    
    for idx, article in enumerate(articles):
        print(f"\n📄 {idx+1}/{len(articles)}: {article['title'][:50]}...")
        
        # Analyze relevance
        score = analyze_relevance(article['title'], article['content'])
        print(f"   🎯 Relevance: {score}/10")
        
        # Only process relevant articles
        if score >= 4:
            summary = generate_summary(article['title'], article['content'])
            mcq = generate_mcq(article['title'], article['content'])
            
            processed.append({
                'Date': article['date'],
                'Title': article['title'],
                'Source': article['source'],
                'Summary': summary,
                'MCQ': mcq,
                'Link': article['link'],
                'Relevance_Score': score
            })
            print(f"   ✅ Processed successfully")
        else:
            print(f"   ⏭️ Skipped (low relevance)")
    
    return processed

# ============================================
# 📊 GOOGLE SHEETS INTEGRATION
# ============================================

def save_to_google_sheets(articles):
    """Save articles to Google Sheets"""
    if not GOOGLE_CREDENTIALS_JSON:
        print("❌ Google credentials not found!")
        return False
    
    try:
        # Parse credentials from JSON string
        creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
        
        # Setup credentials
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        gc = gspread.authorize(credentials)
        
        # Open sheet
        sheet = gc.open_by_key(SHEET_ID).sheet1
        
        # Get existing data to avoid duplicates
        existing_titles = []
        try:
            existing_data = sheet.get_all_records()
            existing_titles = [row.get('Title', '') for row in existing_data]
        except:
            pass
        
        # Append new articles
        new_count = 0
        for article in articles:
            # Check for duplicates
            if article['Title'] in existing_titles:
                print(f"⏭️ Skipping duplicate: {article['Title'][:50]}...")
                continue
            
            row = [
                article['Date'],
                article['Title'],
                article['Source'],
                article['Summary'],
                article['MCQ'],
                article['Link'],
                article['Relevance_Score']
            ]
            
            sheet.append_row(row)
            new_count += 1
            print(f"✅ Saved: {article['Title'][:50]}...")
            time.sleep(1)  # Rate limiting
        
        print(f"\n🎉 Successfully saved {new_count} new articles to Google Sheets!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving to Google Sheets: {e}")
        return False

# ============================================
# 🚀 MAIN EXECUTION
# ============================================

def main():
    print("="*60)
    print("🤖 DAILY NEWS AUTOMATION STARTED")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Fetch news
    print("\n📡 STEP 1: Fetching news...")
    articles = fetch_newsdata_articles()
    
    if not articles:
        print("⚠️ NewsData.io failed, trying RSS feeds...")
        articles = fetch_rss_feeds()
    
    if not articles:
        print("❌ No articles fetched. Exiting.")
        return
    
    print(f"\n✅ Total articles fetched: {len(articles)}")
    
    # Step 2: Process with AI
    print("\n🤖 STEP 2: Processing with AI...")
    processed = process_articles(articles)
    
    if not processed:
        print("⚠️ No articles passed relevance filter.")
        return
    
    print(f"\n✅ Processed {len(processed)} relevant articles")
    
    # Step 3: Save to Google Sheets
    print("\n📊 STEP 3: Saving to Google Sheets...")
    success = save_to_google_sheets(processed)
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 AUTOMATION COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️ AUTOMATION COMPLETED WITH ERRORS")
    print(f"📊 Articles fetched: {len(articles)}")
    print(f"✅ Articles processed: {len(processed)}")
    print(f"⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
