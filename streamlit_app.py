# Debug imports - this will show which package fails
import sys

try:
    import streamlit as st
    st.success("✅ Streamlit loaded")
except ImportError as e:
    print(f"❌ Streamlit failed: {e}")
    sys.exit(1)

try:
    import feedparser
    st.success("✅ Feedparser loaded")
except ImportError as e:
    st.error(f"❌ Feedparser failed: {e}")
    st.stop()

try:
    import pandas as pd
    st.success("✅ Pandas loaded")
except ImportError as e:
    st.error(f"❌ Pandas failed: {e}")
    st.stop()

try:
    from datetime import datetime
    st.success("✅ Datetime loaded")
except ImportError as e:
    st.error(f"❌ Datetime failed: {e}")
    st.stop()

try:
    import google.generativeai as genai
    st.success("✅ Google Generative AI loaded")
except ImportError as e:
    st.error(f"❌ Google Generative AI failed: {e}")
    st.info("💡 Try: pip install google-generativeai")
    st.stop()

import time
import json

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
        errors.append("❌ **Gemini API Key is missing!**\n   - Add it to `.streamlit/secrets.toml` like this:\n   ```\n   GEMINI_API_KEY = 'your-key-here'\n   ```")
    
    if "SHEET_ID" not in st.secrets:
        errors.append("❌ **Google Sheet ID is missing!**\n   - Add it to `.streamlit/secrets.toml` like this:\n   ```\n   SHEET_ID = 'your-sheet-id-here'\n   ```")
    
    if errors:
        st.error("### ⚠️ Setup Not Complete!")
        for error in errors:
            st.markdown(error)
        st.info("💡 **Need help?** Check the setup guide in the README file!")
        st.stop()

check_setup()

# Configure Gemini
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Failed to configure Gemini AI: {e}")
    st.stop()

# RSS Feed URLs
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
        
        # If sheet is empty or only has headers, return empty DataFrame
        if df.empty or len(df) == 0:
            return pd.DataFrame(columns=['Date', 'Title', 'Source', 'Summary', 'MCQ', 'Link'])
        
        return df
    except Exception as e:
        st.warning(f"Could not load data from Google Sheets: {e}")
        return pd.DataFrame(columns=['Date', 'Title', 'Source', 'Summary', 'MCQ', 'Link'])

def save_news_to_sheet(new_data):
    """
    Save news to Google Sheets
    Note: For simplicity, this displays instructions to manually copy-paste
    For automated writing, you'd need Google Sheets API with service account
    """
    st.info("""
    ### 📝 Manual Save Required
    
    To save this data to your Google Sheet:
    1. Copy the table below
    2. Go to your Google Sheet
    3. Paste it as new rows
    
    *(For automatic saving, you'd need to set up Google Sheets API credentials)*
    """)
    
    st.dataframe(new_data)
    
    # Provide CSV download as backup
    csv = new_data.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"news_backup_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================
# 🤖 AI PROCESSING FUNCTIONS
# ============================================

def generate_summary(title, content):
    """Generate UPSC-focused summary using Gemini"""
    prompt = f"""
    You are an expert tutor for Indian competitive exams (UPSC/SSC).
    
    Article Title: {title}
    Article Content: {content[:1000]}
    
    Write a concise 3-4 line summary focusing on:
    - Key facts and figures
    - Relevance to UPSC/SSC syllabus (mention the subject area like Polity, Economy, Geography, etc.)
    - Why this matters for exam preparation
    
    Keep it clear and exam-focused.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Summary generation failed: {e}"

def generate_mcq(title, content):
    """Generate MCQ question using Gemini"""
    prompt = f"""
    You are an expert MCQ creator for UPSC/SSC exams.
    
    Article Title: {title}
    Article Content: {content[:1000]}
    
    Create ONE multiple-choice question in this EXACT format:
    
    Q: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [A/B/C/D]
    Explanation: [1-2 line explanation]
    
    Make it moderately challenging and exam-relevant.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"MCQ generation failed: {e}"

# ============================================
# 📡 NEWS FETCHING FUNCTIONS
# ============================================

def fetch_rss_feeds():
    """Fetch latest news from RSS feeds"""
    all_articles = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (source, url) in enumerate(RSS_FEEDS.items()):
        status_text.text(f"Fetching from {source}...")
        
        try:
            feed = feedparser.parse(url)
            
            # Get top 5 articles from each source
            for entry in feed.entries[:5]:
                article = {
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'title': entry.get('title', 'No title'),
                    'source': source,
                    'link': entry.get('link', ''),
                    'content': entry.get('summary', entry.get('description', ''))[:500]
                }
                all_articles.append(article)
            
            time.sleep(1)  # Be nice to the servers
            
        except Exception as e:
            st.warning(f"Could not fetch from {source}: {e}")
        
        progress_bar.progress((idx + 1) / len(RSS_FEEDS))
    
    status_text.text("✅ Fetching complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return all_articles

def process_articles_with_ai(articles):
    """Process articles with AI to generate summaries and MCQs"""
    processed = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, article in enumerate(articles):
        status_text.text(f"Processing article {idx+1}/{len(articles)}: {article['title'][:50]}...")
        
        try:
            summary = generate_summary(article['title'], article['content'])
            time.sleep(1)  # Rate limiting
            
            mcq = generate_mcq(article['title'], article['content'])
            time.sleep(1)  # Rate limiting
            
            processed.append({
                'Date': article['date'],
                'Title': article['title'],
                'Source': article['source'],
                'Summary': summary,
                'MCQ': mcq,
                'Link': article['link']
            })
            
        except Exception as e:
            st.warning(f"Processing failed for '{article['title'][:30]}...': {e}")
        
        progress_bar.progress((idx + 1) / len(articles))
    
    status_text.text("✅ AI processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(processed)

# ============================================
# 🎨 UI COMPONENTS
# ============================================

def display_news_card(row):
    """Display a single news card"""
    with st.container():
        st.markdown(f"### 📰 {row['Title']}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Source:** {row['Source']} | **Date:** {row['Date']}")
            st.markdown(f"**Summary:**\n{row['Summary']}")
            
            if st.button(f"🔗 Read Full Article", key=f"link_{row.name}"):
                st.markdown(f"[Open in new tab]({row['Link']})")
        
        with col2:
            st.markdown("**🎯 Practice MCQ**")
            with st.expander("Show Question"):
                st.markdown(row['MCQ'])
        
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
        
        if st.button("🔄 Fetch New Articles", type="primary"):
            with st.spinner("Fetching and processing news..."):
                # Fetch articles
                articles = fetch_rss_feeds()
                
                if articles:
                    st.success(f"Fetched {len(articles)} articles!")
                    
                    # Process with AI
                    processed_df = process_articles_with_ai(articles)
                    
                    if not processed_df.empty:
                        st.success(f"Processed {len(processed_df)} articles!")
                        
                        # Store in session state
                        st.session_state['new_articles'] = processed_df
                        
                        # Show save instructions
                        save_news_to_sheet(processed_df)
                    else:
                        st.error("AI processing failed!")
                else:
                    st.error("No articles fetched!")
        
        st.divider()
        st.markdown("""
        ### 📚 About
        This app fetches daily news from:
        - The Hindu
        - Press Information Bureau (PIB)
        
        **Features:**
        - AI-generated summaries
        - Practice MCQs
        - Exam-focused analysis
        """)
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["📋 Daily Feed", "💬 Ask the AI"])
    
    with tab1:
        st.header("Today's Current Affairs")
        
        # Load existing news
        df = load_news_from_sheet()
        
        if df.empty:
            st.info("👋 No articles yet! Click **'Fetch New Articles'** in the sidebar to get started.")
        else:
            # Filter options
            col1, col2 = st.columns([2, 1])
            with col1:
                source_filter = st.multiselect(
                    "Filter by source:",
                    options=df['Source'].unique() if 'Source' in df.columns else [],
                    default=df['Source'].unique() if 'Source' in df.columns else []
                )
            
            with col2:
                st.metric("Total Articles", len(df))
            
            # Filter dataframe
            if source_filter:
                df = df[df['Source'].isin(source_filter)]
            
            # Display news cards
            if df.empty:
                st.warning("No articles match your filters.")
            else:
                for idx, row in df.iterrows():
                    display_news_card(row)
    
    with tab2:
        st.header("💬 Ask Questions About the News")
        
        # Load news for context
        df = load_news_from_sheet()
        
        if df.empty:
            st.info("No articles loaded yet. Fetch some news first!")
        else:
            # Create context from all summaries
            context = "\n\n".join([
                f"Article: {row['Title']}\nSummary: {row['Summary']}"
                for _, row in df.iterrows()
            ])
            
            # Chat interface
            user_question = st.text_input(
                "Ask me anything about today's news:",
                placeholder="E.g., What are the key economic developments today?"
            )
            
            if user_question:
                with st.spinner("Thinking..."):
                    prompt = f"""
                    You are an expert UPSC/SSC tutor. Based on today's news articles, answer this question:
                    
                    Question: {user_question}
                    
                    Context (Today's News):
                    {context[:3000]}
                    
                    Provide a clear, exam-focused answer. If the answer isn't in today's news, say so and provide general knowledge if relevant.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        st.markdown("### 🤖 Answer:")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Could not generate answer: {e}")

# ============================================
# 🚀 RUN THE APP
# ============================================

if __name__ == "__main__":
    main()
