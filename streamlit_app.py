import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import time
import json
import requests
import re

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
        errors.append("❌ **Gemini API Key is missing!**\n   - Add it to Streamlit secrets")
    
    if "SHEET_ID" not in st.secrets:
        errors.append("❌ **Google Sheet ID is missing!**\n   - Add it to Streamlit secrets")
    
    if "NEWSDATA_API_KEY" not in st.secrets:
        errors.append("⚠️ **NewsData.io API Key is missing!**\n   - Add it to Streamlit secrets\n   - Get it from: https://newsdata.io/")
    
    if errors:
        st.error("### ⚠️ Setup Not Complete!")
        for error in errors:
            st.markdown(error)
        st.info("💡 **Add your API keys in:** Streamlit Cloud → App Settings → Secrets")
        st.stop()

check_setup()

# ============================================
# 🔧 ROBUST GEMINI CONFIGURATION (Auto-Detect)
# ============================================

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Show API key preview
    api_key_preview = st.secrets["GEMINI_API_KEY"][:10] + "..."
    st.sidebar.info(f"🔑 Gemini Key: {api_key_preview}")
    
    # Function to find a working model
    def find_working_model(model_candidates, role_name):
        st.sidebar.markdown(f"**🔍 Testing models for {role_name}...**")
        for model_name in model_candidates:
            try:
                # 1. Initialize
                model = genai.GenerativeModel(model_name)
                
                # 2. TEST connection (This is the critical step you were missing)
                response = model.generate_content("Test", request_options={'timeout': 5})
                if response:
                    st.sidebar.success(f"✅ {role_name}: {model_name} (Connected)")
                    return model
            except Exception as e:
                # Silently fail and try the next one
                continue
        
        st.sidebar.error(f"❌ No working model found for {role_name}")
        return None

    # 1. Setup PRIMARY Model (Bulk/Fast)
    # We try generic names first, they are usually safer than specific versions like -002
    primary_candidates = [
        "gemini-1.5-flash",          # Most common stable alias
        "gemini-1.5-flash-latest",   # Rolling update
        "gemini-1.5-pro",            # Fallback to Pro if Flash fails
        "gemini-pro",                # Old faithful 1.0 Pro
        "gemini-1.0-pro"             # Explicit 1.0
    ]
    
    st.session_state.primary_model = find_working_model(primary_candidates, "Primary (Bulk)")
    
    if not st.session_state.primary_model:
        st.error("CRITICAL: Could not find ANY working Gemini model. Check your API Key permissions.")
        st.stop()

    # 2. Setup PREMIUM Model (Chat/Smart)
    # Try the newest, smartest models first
    premium_candidates = [
        "gemini-2.0-flash-exp",      # Bleeding edge
        "gemini-1.5-pro",            # Standard high-intelligence
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash"           # Fallback to Flash if Pro fails
    ]
    
    st.session_state.premium_model = find_working_model(premium_candidates, "Premium (Chat)")
    
    # Fallback: If premium fails, just use the primary model
    if not st.session_state.premium_model:
        st.sidebar.warning("⚠️ Premium model failed. Using Primary for everything.")
        st.session_state.premium_model = st.session_state.primary_model

except Exception as e:
    st.error(f"Failed to configure Gemini AI: {e}")
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
# 📰 NEWS FETCHING FUNCTIONS (NewsData.io)
# ============================================

def fetch_newsdata_articles():
    """Fetch news from NewsData.io API"""
    try:
        api_key = st.secrets["NEWSDATA_API_KEY"]
        
        # Indian news sources - focusing on current affairs
        url = "https://newsdata.io/api/1/news"
        params = {
            'apikey': api_key,
            'country': 'in',
            'language': 'en',
            'category': 'politics,top,world',
            'size': 10  # Get 10 articles per request
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if API returned success
            if data.get('status') == 'success' and 'results' in data:
                articles = []
                for article in data.get('results', []):
                    # Get description or content, ensure it's not None
                    description = article.get('description') or article.get('content') or ''
                    content = description[:800] if description else 'No content available'
                    
                    articles.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'title': article.get('title', 'No title'),
                        'source': (article.get('source_id') or 'Unknown').title(),
                        'link': article.get('link', ''),
                        'content': content
                    })
                
                if articles:
                    return articles
                else:
                    st.warning("No articles found in NewsData.io response")
                    return []
            else:
                # Handle API errors
                error_msg = data.get('results', {}).get('message') if isinstance(data.get('results'), dict) else 'Unknown error'
                st.error(f"NewsData API error: {error_msg}")
                st.info("💡 Tip: Check your API key at https://newsdata.io/")
                return []
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            st.info(f"Response: {response.text[:200]}")
            return []
            
    except Exception as e:
        st.error(f"Error fetching from NewsData.io: {e}")
        st.info("💡 Try unchecking 'Use NewsData.io API' to use RSS feeds instead")
        return []

def fetch_rss_feeds():
    """Fetch news from RSS feeds (backup method)"""
    all_articles = []
    
    for source, url in RSS_FEEDS.items():
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
            
            time.sleep(1)
            
        except Exception as e:
            st.warning(f"Could not fetch from {source}: {e}")
    
    return all_articles

# ============================================
# 🤖 AI PROCESSING FUNCTIONS
# ============================================

def analyze_relevance(title, content):
    """AI analyzes article relevance for UPSC/SSC (score 1-10)"""
    prompt = f"""
    You are an expert UPSC/SSC exam analyzer.
    
    Article Title: {title}
    Article Content: {content[:500]}
    
    Rate this article's relevance for UPSC/SSC exam preparation on a scale of 1-10, where:
    - 10 = Extremely relevant (directly covers syllabus topics, current affairs that will likely appear in exams)
    - 7-9 = Highly relevant (important current affairs, policy changes, international relations)
    - 4-6 = Moderately relevant (general awareness, background knowledge)
    - 1-3 = Low relevance (entertainment, sports, local news)
    
    Respond with ONLY a number from 1-10, nothing else.
    """
    
    try:
        response = st.session_state.primary_model.generate_content(prompt)
        time.sleep(2)  # Rate limiting
        score_text = response.text.strip()
        score = int(''.join(filter(str.isdigit, score_text)))
        return min(max(score, 1), 10)  # Ensure between 1-10
    except Exception as e:
        time.sleep(1)
        return 5  # Default middle score if analysis fails

def generate_summary(title, content):
    """Generate UPSC-focused summary using Gemini - Uses PRIMARY model"""
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
        response = st.session_state.primary_model.generate_content(prompt)
        time.sleep(2)  # Rate limiting
        return response.text.strip()
    except Exception as e:
        time.sleep(2)
        return f"Summary generation failed: {str(e)}"

def generate_mcq(title, content):
    """Generate MCQ question using Gemini - Uses PRIMARY model - RETURNS JSON"""
    # Force JSON output to fix parsing errors
    prompt = f"""
    You are an expert MCQ creator for UPSC/SSC exams.
    
    Article Title: {title}
    Article Content: {content[:1000]}
    
    Create ONE multiple-choice question.
    RETURN ONLY RAW JSON. Do not use Markdown. Do not say "Here is the JSON".
    
    Structure:
    {{
        "question": "The question text here?",
        "options": {{
            "A": "Option A text",
            "B": "Option B text",
            "C": "Option C text",
            "D": "Option D text"
        }},
        "correct": "A",
        "explanation": "Explanation here"
    }}
    """
    
    try:
        # Requesting JSON response via prompt engineering
        response = st.session_state.primary_model.generate_content(prompt)
        time.sleep(2)  # Rate limiting
        return response.text.strip()
    except Exception as e:
        time.sleep(2)
        return json.dumps({"error": str(e)})

def process_articles_with_ai(articles):
    """Process articles with AI: filter by relevance, generate summaries and MCQs"""
    
    # Step 1: Analyze relevance
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🤖 AI analyzing article relevance...")
    st.info("⏱️ This will take ~3 minutes due to API rate limits (5 requests/min). Please be patient!")
    
    for idx, article in enumerate(articles):
        article['relevance_score'] = analyze_relevance(article['title'], article['content'])
        progress_bar.progress((idx + 1) / len(articles) * 0.3)  # 30% for analysis
    
    # Step 2: Sort by relevance and keep top articles
    articles_sorted = sorted(articles, key=lambda x: x['relevance_score'], reverse=True)
    top_articles = articles_sorted[:5]  # REDUCED to 5 articles to avoid rate limits
    
    status_text.text(f"✅ Selected top {len(top_articles)} most relevant articles (processing with delays for rate limits)")
    time.sleep(2)
    
    # Step 3: Generate summaries and MCQs
    processed = []
    
    for idx, article in enumerate(top_articles):
        status_text.text(f"Processing article {idx+1}/{len(top_articles)}: {article['title'][:50]}... (⏱️ ~20 sec per article)")
        
        try:
            summary = generate_summary(article['title'], article['content'])
            mcq = generate_mcq(article['title'], article['content'])
            
            processed.append({
                'Date': article['date'],
                'Title': article['title'],
                'Source': article['source'],
                'Summary': summary,
                'MCQ': mcq,
                'Link': article['link'],
                'Relevance_Score': article['relevance_score']
            })
            
        except Exception as e:
            st.warning(f"Processing failed for '{article['title'][:30]}...': {e}")
        
        progress_bar.progress(0.3 + (idx + 1) / len(top_articles) * 0.7)  # 70% for processing
    
    status_text.text("✅ Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(processed)

# ============================================
# 🎮 INTERACTIVE MCQ COMPONENT
# ============================================

def parse_mcq(mcq_text):
    """Parse JSON MCQ text into structured format"""
    try:
        # Clean potential markdown from AI response
        cleaned_text = mcq_text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(cleaned_text)
        
        question = data.get('question', 'No question')
        options = data.get('options', {})
        correct = data.get('correct', 'A')
        explanation = data.get('explanation', 'No explanation')
        
        return question, options, correct, explanation
    except Exception as e:
        # Fallback for old text format if loaded from history
        try:
            lines = mcq_text.strip().split('\n')
            question = ""
            options = {}
            correct = ""
            explanation = ""
            for line in lines:
                if line.startswith('Q:'): question = line[2:].strip()
                elif line.startswith('A)'): options['A'] = line[2:].strip()
                elif line.startswith('B)'): options['B'] = line[2:].strip()
                elif line.startswith('C)'): options['C'] = line[2:].strip()
                elif line.startswith('D)'): options['D'] = line[2:].strip()
                elif 'CORRECT:' in line: correct = line.split('CORRECT:')[1].strip()[0]
                elif 'EXPLANATION:' in line: explanation = line.split(':')[1].strip()
            
            if question and options:
                return question, options, correct, explanation
            return None, None, None, None
        except:
            return None, None, None, None

def display_interactive_mcq(mcq_text, article_index):
    """Display interactive MCQ with buttons and feedback"""
    
    question, options, correct_answer, explanation = parse_mcq(mcq_text)
    
    if not question or not options:
        st.warning("⚠️ Could not parse MCQ properly")
        with st.expander("📄 Show Raw MCQ"):
            st.text(mcq_text)
        return
    
    # Create unique key for this MCQ
    mcq_key = f"mcq_{article_index}"
    answer_key = f"answer_{article_index}"
    
    # Initialize session state for this MCQ
    if answer_key not in st.session_state:
        st.session_state[answer_key] = None
    
    # Display question
    st.markdown(f"**❓ Question:**")
    st.markdown(f"*{question}*")
    st.markdown("")
    
    # Display options as buttons
    cols = st.columns(2)
    
    option_labels = ['A', 'B', 'C', 'D']
    for idx, label in enumerate(option_labels):
        if label in options:
            col = cols[idx % 2]
            with col:
                # Determine button styling
                if st.session_state[answer_key] == label:
                    if label == correct_answer:
                        button_type = "primary"
                        emoji = "✅"
                    else:
                        button_type = "secondary"
                        emoji = "❌"
                    button_text = f"{emoji} {label}) {options[label]}"
                else:
                    button_type = "secondary"
                    emoji = ""
                    button_text = f"{label}) {options[label]}"
                
                if st.button(
                    button_text,
                    key=f"{mcq_key}_{label}",
                    disabled=st.session_state[answer_key] is not None,
                    use_container_width=True,
                    type=button_type if st.session_state[answer_key] == label else "secondary"
                ):
                    st.session_state[answer_key] = label
                    st.rerun()
    
    # Show feedback after answer
    if st.session_state[answer_key] is not None:
        st.markdown("---")
        
        if st.session_state[answer_key] == correct_answer:
            st.success(f"✅ **Correct!** Well done!")
        else:
            st.error(f"❌ **Wrong!** The correct answer is **{correct_answer}**")
        
        if explanation:
            st.info(f"💡 **Explanation:** {explanation}")
        
        # Reset button
        if st.button("🔄 Try Another Question", key=f"reset_{mcq_key}"):
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
        display_interactive_mcq(row['MCQ'], index)
        
        st.divider()

# ============================================
# 🎯 MAIN APP
# ============================================

def main():
    # Header
    st.title("📰 Current Affairs Feed for UPSC/SSC")
    st.markdown("*Your AI-powered exam preparation companion with smart filtering*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # News source selector
        st.markdown("### 📡 News Source")
        use_newsdata = st.checkbox("Use NewsData.io API", value=True, help="Uses premium news API for better sources")
        
        if st.button("🔄 Fetch New Articles", type="primary"):
            with st.spinner("Fetching and analyzing news..."):
                # Fetch articles
                if use_newsdata and "NEWSDATA_API_KEY" in st.secrets:
                    st.info("📡 Fetching from NewsData.io (Times of India, Hindu, etc.)...")
                    articles = fetch_newsdata_articles()
                else:
                    st.info("📡 Fetching from RSS feeds...")
                    articles = fetch_rss_feeds()
                
                if articles:
                    st.success(f"Fetched {len(articles)} articles!")
                    
                    # Process with AI (includes smart filtering)
                    processed_df = process_articles_with_ai(articles)
                    
                    if not processed_df.empty:
                        st.success(f"✅ Processed {len(processed_df)} top articles!")
                        
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
        ### 📚 Features
        - 🤖 AI-powered relevance scoring
        - 🎯 Smart article filtering
        - 📝 Exam-focused summaries
        - 🎮 Interactive MCQ practice
        - 📊 Multiple news sources
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
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if 'Source' in df.columns:
                    source_filter = st.multiselect(
                        "Filter by source:",
                        options=df['Source'].unique(),
                        default=df['Source'].unique()
                    )
                else:
                    source_filter = []
            
            with col2:
                st.metric("Total Articles", len(df))
            
            with col3:
                if 'Relevance_Score' in df.columns:
                    avg_score = df['Relevance_Score'].mean()
                    st.metric("Avg Relevance", f"{avg_score:.1f}/10")
            
            # Filter dataframe
            if source_filter and 'Source' in df.columns:
                df = df[df['Source'].isin(source_filter)]
            
            # Sort by relevance score if available
            if 'Relevance_Score' in df.columns:
                df = df.sort_values('Relevance_Score', ascending=False)
            
            # Display news cards
            if df.empty:
                st.warning("No articles match your filters.")
            else:
                for idx, row in df.iterrows():
                    display_news_card(row, idx)
    
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
                with st.spinner("Thinking... (Using premium Gemini 2.5-flash model)"):
                    prompt = f"""
                    You are an expert UPSC/SSC tutor. Based on today's news articles, answer this question:
                    
                    Question: {user_question}
                    
                    Context (Today's News):
                    {context[:3000]}
                    
                    Provide a clear, exam-focused answer. If the answer isn't in today's news, say so and provide general knowledge if relevant.
                    """
                    
                    try:
                        # Use PREMIUM model for personalized questions
                        response = st.session_state.premium_model.generate_content(prompt)
                        st.markdown("### 🤖 Answer:")
                        st.markdown(response.text)
                        st.info("✨ Powered by Gemini 2.5-flash (Premium)")
                    except Exception as e:
                        st.error(f"Could not generate answer: {e}")
                        st.info("💡 Tip: You have 20 premium queries per day. Try again in a few minutes if you hit the limit.")

# ============================================
# 🚀 RUN THE APP
# ============================================

if __name__ == "__main__":
    main()
