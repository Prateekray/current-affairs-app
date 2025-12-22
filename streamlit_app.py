import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import time
import json
import requests
import openai

# ============================================
# 🔧 CONFIGURATION & SETUP
# ============================================

st.set_page_config(
    page_title="Current Affairs Feed - UPSC/SSC",
    page_icon="📰",
    layout="wide"
)

# Check for required secrets
def check_setup():
    """Friendly error messages if setup is incomplete"""
    errors = []
    
    if "GEMINI_API_KEY" not in st.secrets:
        errors.append("❌ **Gemini API Key is missing!** (Required for Chat)\n   - Get it at: https://aistudio.google.com/apikey")
    
    if "GROQ_API_KEY" not in st.secrets:
        errors.append("❌ **Groq API Key is missing!** (Required for Bulk Processing)\n   - Get it free at: https://console.groq.com")
        
    if "SHEET_ID" not in st.secrets:
        errors.append("❌ **Google Sheet ID is missing!**")
    
    if "NEWSDATA_API_KEY" not in st.secrets:
        errors.append("⚠️ **NewsData.io API Key is missing!** (Optional)\n   - Get it at: https://newsdata.io/register")
    
    if errors:
        st.error("### ⚠️ Setup Not Complete!")
        for error in errors:
            st.markdown(error)
        st.info("💡 **Add your API keys in:** Streamlit Cloud → App Settings → Secrets")
        st.stop()

check_setup()

# ============================================
# 🔧 HYBRID AI CONFIGURATION (Groq + Gemini)
# ============================================

try:
    # 1. Setup Groq (Primary/Bulk - FREE & FAST)
    groq_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    st.session_state.groq_client = groq_client
    st.sidebar.success(f"✅ Bulk AI: Groq Llama-3.1 (Turbo)")

    # 2. Setup Gemini (Premium/Chat - SMART)
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Try to find the best Gemini model available
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except:
        pass

    # Prefer newer models for better quality
    model_preferences = [
        "models/gemini-3-pro-preview",
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash-exp",
        "models/gemini-1.5-flash"
    ]
    
    chat_model_name = None
    for preferred in model_preferences:
        if preferred in available_models:
            chat_model_name = preferred
            break
    
    if not chat_model_name:
        chat_model_name = "gemini-1.5-flash"  # Fallback
        
    st.session_state.premium_model = genai.GenerativeModel(chat_model_name)
    model_display = chat_model_name.replace('models/', '')
    st.sidebar.success(f"🌟 Chat AI: {model_display}")

except Exception as e:
    st.error(f"❌ Failed to configure AI: {e}")
    st.info("💡 Check your API keys in Streamlit secrets")
    st.stop()

# RSS Feed URLs (backup)
RSS_FEEDS = {
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss",
    "PIB": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1"
}

# ============================================
# 🗄️ DATABASE FUNCTIONS (Google Sheets)
# ============================================

def get_sheet_url():
    """Generate the public Google Sheets CSV export URL"""
    sheet_id = st.secrets["SHEET_ID"]
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"

def load_news_from_sheet():
    """Load news from Google Sheets"""
    try:
        url = get_sheet_url()
        df = pd.read_csv(url)
        
        if df.empty or len(df) == 0:
            return pd.DataFrame(columns=['Date', 'Title', 'Source', 'Summary', 'MCQ', 'Link', 'Relevance_Score'])
        
        return df
    except Exception as e:
        st.warning(f"Could not load data from Google Sheets: {e}")
        return pd.DataFrame(columns=['Date', 'Title', 'Source', 'Summary', 'MCQ', 'Link', 'Relevance_Score'])

def save_news_to_sheet(new_data):
    """Display instructions to manually save data"""
    st.info("""
    ### 📝 Manual Save Required
    
    To save this data to your Google Sheet:
    1. Copy the table below
    2. Go to your Google Sheet
    3. Paste it as new rows
    """)
    
    st.dataframe(new_data, use_container_width=True)
    
    csv = new_data.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"news_backup_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================
# 📰 NEWS FETCHING FUNCTIONS
# ============================================

def fetch_newsdata_articles():
    """Fetch news from NewsData.io API"""
    if "NEWSDATA_API_KEY" not in st.secrets:
        st.warning("NewsData.io API key not configured. Using RSS feeds instead.")
        return []
    
    try:
        api_key = st.secrets["NEWSDATA_API_KEY"]
        url = "https://newsdata.io/api/1/news"
        params = {
            'apikey': api_key,
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
                    content = description[:800] if description else 'No content available'
                    
                    articles.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'title': article.get('title', 'No title'),
                        'source': (article.get('source_id') or 'Unknown').title(),
                        'link': article.get('link', ''),
                        'content': content
                    })
                return articles
            else:
                error_msg = data.get('results', {}).get('message', 'Unknown error')
                st.warning(f"NewsData.io API returned: {error_msg}")
                return []
        else:
            st.warning(f"NewsData.io API status code: {response.status_code}")
            return []
            
    except Exception as e:
        st.warning(f"Error fetching from NewsData.io: {e}")
        return []

def fetch_rss_feeds():
    """Fetch news from RSS feeds (backup method)"""
    all_articles = []
    
    progress = st.progress(0)
    for idx, (source, url) in enumerate(RSS_FEEDS.items()):
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                article = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'title': entry.get('title', 'No title'),
                    'source': source,
                    'link': entry.get('link', ''),
                    'content': entry.get('summary', entry.get('description', ''))[:800]
                }
                all_articles.append(article)
            time.sleep(1)  # Be nice to servers
        except Exception as e:
            st.warning(f"Could not fetch from {source}: {e}")
        
        progress.progress((idx + 1) / len(RSS_FEEDS))
    
    progress.empty()
    return all_articles

# ============================================
# 🤖 AI PROCESSING FUNCTIONS (GROQ POWERED)
# ============================================

def get_groq_response(system_prompt, user_content, json_mode=False):
    """Helper to call Groq Llama-3 (with retry logic)"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            # Enforce JSON output if requested
            if json_mode:
                system_prompt += " YOU MUST RETURN ONLY VALID JSON. NO MARKDOWN. NO CODE BLOCKS. NO EXPLANATORY TEXT."
                
            response = st.session_state.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Extremely fast & free
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"⚠️ Groq API Error: {e}")
                st.info("💡 Get your free Groq API key at: https://console.groq.com")
                return None
            time.sleep(1)  # Brief pause before retry

def analyze_relevance(title, content):
    """Analyze article relevance using Groq (1-10 score)"""
    prompt = f"Article Title: {title}\n\nContent Preview: {content[:500]}"
    
    system = """You are a UPSC/SSC exam relevance analyzer. 
    
    Rate articles 1-10:
    - 10: Extremely relevant (government policies, international relations, major economic news)
    - 7-9: Highly relevant (important current affairs, constitutional matters)
    - 4-6: Moderately relevant (general awareness topics)
    - 1-3: Low relevance (entertainment, sports, local crimes)
    
    Return ONLY this JSON format: {"score": number}
    Example: {"score": 8}"""
    
    response = get_groq_response(system, prompt, json_mode=True)
    
    try:
        # Clean potential markdown
        cleaned = response.replace("```json", "").replace("```", "").strip()
        score = json.loads(cleaned)['score']
        return min(max(int(score), 1), 10)  # Ensure 1-10 range
    except:
        return 5  # Default to middle score

def generate_summary(title, content):
    """Generate UPSC-focused summary using Groq"""
    prompt = f"Article Title: {title}\n\nContent: {content[:1000]}"
    
    system = """You are an expert UPSC/SSC tutor. 
    
    Write a concise 3-4 line summary that:
    1. Highlights key facts and figures
    2. Mentions the syllabus area (Polity, Economy, Geography, International Relations, etc.)
    3. Explains why this matters for exam preparation
    
    Keep it clear, factual, and exam-focused."""
    
    result = get_groq_response(system, prompt)
    return result or "Summary generation failed."

def generate_mcq(title, content):
    """Generate MCQ using Groq (JSON format)"""
    prompt = f"Article Title: {title}\n\nContent: {content[:1000]}"
    
    system = """You are an expert UPSC/SSC MCQ creator.
    
    Create ONE challenging multiple-choice question based on this article.
    
    Return ONLY this JSON format (no markdown, no code blocks):
    {
        "question": "Your question text here",
        "options": {
            "A": "First option",
            "B": "Second option",
            "C": "Third option",
            "D": "Fourth option"
        },
        "correct": "A",
        "explanation": "Brief explanation why this is correct"
    }
    
    Make it exam-standard difficulty."""
    
    response = get_groq_response(system, prompt, json_mode=True)
    
    try:
        # Clean markdown wrappers
        cleaned = response.replace("```json", "").replace("```", "").strip()
        # Validate it's valid JSON
        json.loads(cleaned)
        return cleaned
    except:
        return json.dumps({
            "question": "Question generation failed",
            "options": {"A": "N/A", "B": "N/A", "C": "N/A", "D": "N/A"},
            "correct": "A",
            "explanation": "Unable to generate MCQ for this article"
        })

def process_articles_with_ai(articles):
    """Process articles with Groq AI (fast bulk processing)"""
    
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("⚡ Groq AI analyzing articles (turbo speed)...")
    
    processed = []
    
    for idx, article in enumerate(articles):
        try:
            # Step 1: Analyze relevance
            score = analyze_relevance(article['title'], article['content'])
            
            # Step 2: Filter - only process relevant articles (score >= 4)
            if score >= 4:
                status_text.text(f"⚡ Processing: {article['title'][:50]}... (Score: {score}/10)")
                
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
            else:
                status_text.text(f"⏭️ Skipping: {article['title'][:50]}... (Score: {score}/10 - Too low)")
        
        except Exception as e:
            st.warning(f"⚠️ Failed to process: {article['title'][:30]}... - {e}")
        
        # Update progress
        progress_bar.progress((idx + 1) / len(articles))
        time.sleep(0.1)  # Brief pause for UI update
    
    # Calculate processing time
    elapsed = time.time() - start_time
    
    status_text.empty()
    progress_bar.empty()
    
    if processed:
        st.success(f"✅ Processed {len(processed)} relevant articles in {elapsed:.1f} seconds! ⚡")
    else:
        st.warning("No articles met the relevance threshold (score >= 4)")
    
    return pd.DataFrame(processed)

# ============================================
# 🎮 INTERACTIVE MCQ COMPONENT
# ============================================

def parse_mcq(mcq_text):
    """Parse JSON MCQ into structured format"""
    try:
        # Clean potential markdown
        cleaned = mcq_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        
        question = data.get('question', 'No question')
        options = data.get('options', {})
        correct = data.get('correct', 'A')
        explanation = data.get('explanation', 'No explanation')
        
        return question, options, correct, explanation
    except:
        return None, None, None, None

def display_interactive_mcq(row, index):
    """Display interactive MCQ with Retry and New Question features"""
    
    # Check for custom generated question
    custom_key = f"mcq_custom_{index}"
    if custom_key in st.session_state:
        mcq_text = st.session_state[custom_key]
    else:
        mcq_text = row['MCQ']
    
    # Parse MCQ
    question, options, correct_answer, explanation = parse_mcq(mcq_text)
    
    if not question or not options:
        st.caption("⚠️ MCQ unavailable for this article")
        return
    
    # Setup state keys
    answer_key = f"answer_{index}"
    
    if answer_key not in st.session_state:
        st.session_state[answer_key] = None
    
    # Display question
    st.markdown(f"**❓ Question:**")
    st.markdown(f"*{question}*")
    st.markdown("")
    
    # Display option buttons
    cols = st.columns(2)
    option_labels = ['A', 'B', 'C', 'D']
    
    for idx, label in enumerate(option_labels):
        if label in options:
            col = cols[idx % 2]
            with col:
                is_selected = (st.session_state[answer_key] == label)
                is_disabled = (st.session_state[answer_key] is not None)
                
                # Button styling
                if is_selected:
                    if label == correct_answer:
                        btn_type = "primary"
                        btn_text = f"✅ {label}) {options[label]}"
                    else:
                        btn_type = "secondary"
                        btn_text = f"❌ {label}) {options[label]}"
                else:
                    btn_type = "secondary"
                    btn_text = f"{label}) {options[label]}"
                
                # Render button
                if st.button(
                    btn_text,
                    key=f"btn_{index}_{label}",
                    disabled=is_disabled,
                    use_container_width=True,
                    type=btn_type
                ):
                    st.session_state[answer_key] = label
                    st.rerun()
    
    # Feedback and action buttons
    if st.session_state[answer_key] is not None:
        st.markdown("---")
        
        # Result message
        if st.session_state[answer_key] == correct_answer:
            st.success("✅ **Correct!** Excellent work!")
        else:
            st.error(f"❌ **Wrong!** The correct answer is **{correct_answer}**")
        
        # Explanation
        if explanation:
            st.info(f"💡 **Explanation:** {explanation}")
        
        # Action buttons
        col_retry, col_new = st.columns(2)
        
        with col_retry:
            if st.button("🔄 Retry This Question", key=f"retry_{index}", use_container_width=True):
                st.session_state[answer_key] = None
                st.rerun()
        
        with col_new:
            if st.button("🎲 Generate New Question", key=f"newq_{index}", use_container_width=True):
                with st.spinner("🤖 AI generating fresh question..."):
                    # Generate new question using title and summary
                    new_mcq = generate_mcq(row['Title'], row['Summary'])
                    st.session_state[custom_key] = new_mcq
                    st.session_state[answer_key] = None
                    st.rerun()

# ============================================
# 🎨 UI COMPONENTS
# ============================================

def display_news_card(row, index):
    """Display a single news card with interactive MCQ"""
    with st.container():
        # Header with relevance score
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### 📰 {row['Title']}")
        with col2:
            if 'Relevance_Score' in row and pd.notna(row['Relevance_Score']):
                score = int(row['Relevance_Score'])
                # Color-coded score
                if score >= 8:
                    st.metric("🎯 Relevance", f"{score}/10", delta="High")
                elif score >= 6:
                    st.metric("🎯 Relevance", f"{score}/10", delta="Medium")
                else:
                    st.metric("🎯 Relevance", f"{score}/10")
        
        # Source and date
        st.markdown(f"**Source:** {row['Source']} | **Date:** {row['Date']}")
        
        # Summary
        st.markdown(f"**📝 Summary:**")
        st.markdown(row['Summary'])
        
        # Link
        st.markdown(f"[🔗 Read Full Article]({row['Link']})")
        
        st.markdown("---")
        
        # Interactive MCQ
        st.markdown("### 🎯 Practice MCQ")
        display_interactive_mcq(row, index)
        
        st.divider()

# ============================================
# 🎯 MAIN APP
# ============================================

def main():
    # Header
    st.title("📰 Current Affairs Feed for UPSC/SSC")
    st.markdown("*Your AI-powered exam preparation companion*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # News source selector
        st.markdown("### 📡 News Source")
        use_newsdata = st.checkbox(
            "Use NewsData.io API", 
            value="NEWSDATA_API_KEY" in st.secrets,
            help="Premium news sources (Times of India, Hindu, etc.)"
        )
        
        if st.button("🔄 Fetch & Process Articles", type="primary"):
            with st.spinner("Fetching news..."):
                # Fetch articles
                if use_newsdata:
                    st.info("📡 Fetching from NewsData.io (premium sources)...")
                    articles = fetch_newsdata_articles()
                    
                    # Fallback to RSS if NewsData fails
                    if not articles:
                        st.info("📡 Falling back to RSS feeds...")
                        articles = fetch_rss_feeds()
                else:
                    st.info("📡 Fetching from RSS feeds (The Hindu, PIB)...")
                    articles = fetch_rss_feeds()
                
                if articles:
                    st.success(f"✅ Fetched {len(articles)} articles!")
                    
                    # Process with Groq AI
                    processed_df = process_articles_with_ai(articles)
                    
                    if not processed_df.empty:
                        # Store in session state
                        st.session_state['new_articles'] = processed_df
                        
                        # Show save instructions
                        save_news_to_sheet(processed_df)
                    else:
                        st.warning("⚠️ No articles passed the relevance filter (score >= 4)")
                else:
                    st.error("❌ No articles fetched. Check your internet connection.")
        
        st.divider()
        
        st.markdown("""
        ### ⚡ Tech Stack
        - **Bulk Processing:** Groq Llama-3.1
          - Unlimited free requests
          - Lightning fast (2-3 sec/article)
        - **Premium Chat:** Google Gemini
          - Highest quality responses
          - Context-aware answers
        
        ### 📊 Features
        - Smart relevance filtering
        - Interactive MCQ practice
        - Retry & new question options
        - Multi-source news aggregation
        """)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📋 Daily Feed", "💬 Ask the AI"])
    
    with tab1:
        st.header("📚 Today's Current Affairs")
        
        # Load existing news
        df = load_news_from_sheet()
        
        if df.empty:
            st.info("👋 **No articles yet!**\n\nClick **'Fetch & Process Articles'** in the sidebar to get started.")
            st.markdown("---")
            st.markdown("### 🚀 How it works:")
            st.markdown("""
            1. **Fetch** - Gets latest news from multiple sources
            2. **Analyze** - AI scores each article for UPSC/SSC relevance
            3. **Filter** - Only shows high-quality, exam-relevant content
            4. **Practice** - Interactive MCQs with instant feedback
            """)
        else:
            # Filter controls
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if 'Source' in df.columns:
                    sources = df['Source'].unique().tolist()
                    source_filter = st.multiselect(
                        "Filter by source:",
                        options=sources,
                        default=sources
                    )
                else:
                    source_filter = []
            
            with col2:
                st.metric("📄 Total Articles", len(df))
            
            with col3:
                if 'Relevance_Score' in df.columns and not df['Relevance_Score'].isna().all():
                    avg_score = df['Relevance_Score'].mean()
                    st.metric("⭐ Avg Relevance", f"{avg_score:.1f}/10")
            
            # Apply filters
            if source_filter and 'Source' in df.columns:
                df = df[df['Source'].isin(source_filter)]
            
            # Sort by relevance
            if 'Relevance_Score' in df.columns:
                df = df.sort_values('Relevance_Score', ascending=False)
            
            # Display articles
            if df.empty:
                st.warning("No articles match your selected filters.")
            else:
                for idx, row in df.iterrows():
                    display_news_card(row, idx)
    
    with tab2:
        st.header("💬 Ask AI About the News")
        st.markdown("*Powered by Google Gemini (Premium)*")
        
        # Load news for context
        df = load_news_from_sheet()
        
        if df.empty:
            st.info("📚 No articles loaded yet.\n\nFetch some news first to ask questions!")
        else:
            # Create context from summaries
            context = "\n\n".join([
                f"**{row['Title']}**\n{row['Summary']}"
                for _, row in df.iterrows()
            ])
            
            # Chat interface
            st.markdown("### 🗨️ Your Question")
            user_question = st.text_area(
                "Ask anything about today's current affairs:",
                placeholder="Example: What are the key government policies mentioned today?\n\nOr: Explain the economic implications of today's news.",
                height=100
            )
            
            if st.button("🚀 Get Answer", type="primary"):
                if not user_question.strip():
                    st.warning("Please enter a question first!")
                else:
                    with st.spinner("🤖 Gemini AI thinking..."):
                        prompt = f"""You are an expert UPSC/SSC tutor. Answer this question based on today's current affairs:

**Question:** {user_question}

**Today's News Context:**
{context[:5000]}

Provide a clear, exam-focused answer. If the question relates to current affairs, use the context above. If it's a general knowledge question, provide accurate information and relate it to exam preparation where possible."""
                        
                        try:
                            response = st.session_state.premium_model.generate_content(prompt)
                            
                            st.markdown("---")
                            st.markdown("### 🤖 AI Answer:")
                            st.markdown(response.text)
                            st.caption("✨ Powered by Google Gemini (Premium AI)")
                            
                        except Exception as e:
                            st.error(f"❌ Error generating response: {e}")
                            st.info("💡 Try rephrasing your question or check your Gemini API key.")

# ============================================
# 🚀 RUN THE APP
# ============================================

if __name__ == "__main__":
    main()
