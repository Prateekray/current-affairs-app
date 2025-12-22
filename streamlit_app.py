import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import time
import json
import requests
import openai  # Required for Groq
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
        errors.append("❌ **Gemini API Key is missing!** (Required for Chat)")
    
    if "GROQ_API_KEY" not in st.secrets:
        errors.append("❌ **Groq API Key is missing!** (Required for Bulk Processing)\n   - Get it free at: https://console.groq.com")
        
    if "SHEET_ID" not in st.secrets:
        errors.append("❌ **Google Sheet ID is missing!**")
    
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
    # We use the OpenAI client but point it to Groq's URL
    groq_client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    st.sidebar.success(f"✅ Primary (Bulk): Groq Llama-3 (Turbo)")

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

    # Prefer Gemini 3 or 2.5 Pro for chat
    if "models/gemini-3-pro-preview" in available_models:
        chat_model_name = "gemini-3-pro-preview"
    elif "models/gemini-2.5-pro" in available_models:
        chat_model_name = "gemini-2.5-pro"
    elif "models/gemini-2.0-flash-exp" in available_models:
        chat_model_name = "gemini-2.0-flash-exp"
    else:
        chat_model_name = "gemini-1.5-flash" # Fallback
        
    st.session_state.premium_model = genai.GenerativeModel(chat_model_name)
    st.sidebar.success(f"🌟 Premium (Chat): {chat_model_name.replace('models/', '')}")

except Exception as e:
    st.error(f"Failed to configure AI: {e}")
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
        
        response = requests.get(url, params=params)
        
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
                return []
        else:
            return []
    except Exception as e:
        st.error(f"Error fetching from NewsData.io: {e}")
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
        except Exception as e:
            st.warning(f"Could not fetch from {source}: {e}")
    return all_articles

# ============================================
# 🤖 AI PROCESSING FUNCTIONS (GROQ POWERED)
# ============================================

def get_groq_response(system_prompt, user_content, json_mode=False):
    """Helper to call Groq Llama-3"""
    try:
        # If JSON mode is requested, we strictly enforce it in the prompt
        if json_mode:
            system_prompt += " RETURN ONLY RAW JSON. NO MARKDOWN. NO CODE BLOCKS."
            
        response = groq_client.chat.completions.create(
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
        st.error(f"Groq Error: {e}")
        return None

def analyze_relevance(title, content):
    """Analyze relevance using Groq"""
    prompt = f"Article: {title}\nContent: {content[:500]}"
    system = """You are a UPSC exam analyzer. Rate relevance 1-10. 
    Return JSON format: {"score": number}"""
    
    res = get_groq_response(system, prompt, json_mode=True)
    try:
        # Clean potential markdown wrappers
        clean_res = res.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_res)['score']
    except:
        return 5

def generate_summary(title, content):
    """Generate Summary using Groq"""
    prompt = f"Article: {title}\nContent: {content[:1000]}"
    system = """You are a UPSC tutor. Write a 3-4 line summary focusing on exam relevance.
    Mention key facts and syllabus area (Polity, Economy, etc)."""
    
    return get_groq_response(system, prompt) or "Summary generation failed."

def generate_mcq(title, content):
    """Generate MCQ using Groq"""
    prompt = f"Article: {title}\nContent: {content[:1000]}"
    system = """Create one UPSC-style MCQ. 
    Return JSON format:
    {
        "question": "Question text",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "correct": "A",
        "explanation": "..."
    }"""
    
    res = get_groq_response(system, prompt, json_mode=True)
    try:
        # Robust cleaning of markdown
        clean_res = res.replace("```json", "").replace("```", "").strip()
        # Verify it parses
        json.loads(clean_res) 
        return clean_res
    except:
        return json.dumps({"error": "Failed to generate valid JSON"})

def process_articles_with_ai(articles):
    """Process all articles using Groq"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("⚡ Groq processing (Turbo Speed)...")
    
    processed = []
    
    for idx, article in enumerate(articles):
        # 1. Relevance
        score = analyze_relevance(article['title'], article['content'])
        
        # Smart Filter: Only keep relevant news (>4)
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
        
        # Update progress
        progress_bar.progress((idx + 1) / len(articles))
    
    status_text.success("✅ Processing complete!")
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
        return None, None, None, None

def display_interactive_mcq(mcq_text, article_index):
    """Display interactive MCQ with buttons and feedback"""
    
    question, options, correct_answer, explanation = parse_mcq(mcq_text)
    
    if not question or not options:
        # Silently fail or show minimal error if parsing fails
        if mcq_text and "error" in mcq_text:
            st.caption("⚠️ MCQ generation skipped for this article.")
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
                    st.info("📡 Fetching from NewsData.io...")
                    articles = fetch_newsdata_articles()
                else:
                    st.info("📡 Fetching from RSS feeds...")
                    articles = fetch_rss_feeds()
                
                if articles:
                    st.success(f"Fetched {len(articles)} articles!")
                    
                    # Process with Groq AI
                    processed_df = process_articles_with_ai(articles)
                    
                    if not processed_df.empty:
                        st.success(f"✅ Processed {len(processed_df)} relevant articles!")
                        
                        # Store in session state
                        st.session_state['new_articles'] = processed_df
                        
                        # Show save instructions
                        save_news_to_sheet(processed_df)
                    else:
                        st.warning("No relevant articles found (Score >= 4).")
                else:
                    st.error("No articles fetched!")
        
        st.divider()
        st.markdown("""
        ### 📚 Tech Stack
        - **Bulk AI:** Groq Llama-3 (Fast & Free)
        - **Chat AI:** Gemini 3 Pro (Smart)
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
                with st.spinner("Thinking... (Using Gemini 3 Pro)"):
                    prompt = f"""
                    You are an expert UPSC/SSC tutor. Based on today's news articles, answer this question:
                    
                    Question: {user_question}
                    
                    Context (Today's News):
                    {context[:5000]}
                    
                    Provide a clear, exam-focused answer.
                    """
                    
                    try:
                        # Use PREMIUM Gemini model for personalized questions
                        response = st.session_state.premium_model.generate_content(prompt)
                        st.markdown("### 🤖 Answer:")
                        st.markdown(response.text)
                        st.caption("✨ Powered by Google Gemini (Premium Tier)")
                    except Exception as e:
                        st.error(f"Could not generate answer: {e}")

# ============================================
# 🚀 RUN THE APP
# ============================================

if __name__ == "__main__":
    main()
