import streamlit as st
import requests
import os
from urllib.parse import urlparse

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Perplexity AI Clone",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# SESSION STATE
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "Automatic"
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = ""
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "product_ideas" not in st.session_state:
    st.session_state.product_ideas = []

# =====================================
# CONFIGURATION
# =====================================
# Use environment variable for Azure deployment, fallback to localhost for local dev
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
WORKSPACE = "default"

# MODE MAPPING - All 8 modes with correct backend endpoints
MODES = {
    "Automatic": {
        "icon": "🔍",
        "desc": "Auto-routes to best mode",
        "endpoint": "/api/chat"
    },
    "Web Search": {
        "icon": "🌐",
        "desc": "Real-time web search",
        "endpoint": "/api/web"
    },
    "RAG": {
        "icon": "📚",
        "desc": "Search uploaded documents",
        "endpoint": "/api/rag"
    },
    "Agentic": {
        "icon": "🤖",
        "desc": "Multi-agent collaboration",
        "endpoint": "/api/agentic"
    },
    "Deep Research": {
        "icon": "🧠",
        "desc": "In-depth research",
        "endpoint": "/api/deep_research"
    },
    "Analysis": {
        "icon": "📊",
        "desc": "Deep data analysis",
        "endpoint": "/api/analyze"
    },
    "Summarize": {
        "icon": "📝",
        "desc": "Summarize content",
        "endpoint": "/api/summarize"
    },
    "Chat": {
        "icon": "💬",
        "desc": "Direct AI chat",
        "endpoint": "/api/focus"
    },
    "Product MVP": {
        "icon": "🚀",
        "desc": "Idea → MVP Blueprint",
        "endpoint": "/api/product_mvp"
    },
    "Video Brain": {
        "icon": "🎥",
        "desc": "Understand YouTube lectures",
        "endpoint": "/api/video_brain"
    },
}

# =====================================
# CSS - PERPLEXITY EXACT STYLE
# =====================================
def get_css():
    is_dark = st.session_state.theme == "dark"
    
    if is_dark:
        colors = {
            "bg": "#191A1A",
            "bg2": "#1F2020", 
            "bg3": "#2A2B2B",
            "text": "#ECECEC",
            "text2": "#A1A1A1",
            "muted": "#6B6B6B",
            "accent": "#20B8CD",
            "border": "#3A3B3B",
            "success": "#22C55E"
        }
    else:
        colors = {
            "bg": "#FFFFFF",
            "bg2": "#F7F7F8",
            "bg3": "#EEEEEF",
            "text": "#1A1A1A",
            "text2": "#666666",
            "muted": "#999999",
            "accent": "#0EA5E9",
            "border": "#E5E5E5",
            "success": "#22C55E"
        }
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    * {{ font-family: 'Inter', sans-serif !important; }}
    
    #MainMenu, footer, header, [data-testid="stToolbar"], .stDeployButton {{ display: none !important; }}
    
    .stApp {{ background: {colors['bg']} !important; }}
    
    [data-testid="stSidebar"] {{
        background: {colors['bg']} !important;
        border-right: 1px solid {colors['border']} !important;
    }}
    
    /* Hero */
    .hero {{
        text-align: center;
        padding: 30px 0 15px;
    }}
    .hero-compact {{
        text-align: center;
        padding: 15px 0 10px;
    }}
    .hero-compact .logo {{
        font-size: 28px;
    }}
    .hero-compact .tagline {{
        display: none;
    }}
    .logo {{
        font-size: 40px;
        font-weight: 600;
        color: {colors['text']};
        letter-spacing: -1px;
    }}
    .logo span {{
        background: linear-gradient(135deg, {colors['accent']}, #14B8A6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .tagline {{
        color: {colors['muted']};
        font-size: 14px;
        margin-top: 5px;
    }}
    
    /* UNIFIED SEARCH BOX - All elements inside */
    .search-wrapper {{
        max-width: 800px;
        margin: 0 auto;
        padding: 0 20px;
    }}
    
    /* Hide streamlit defaults */
    .stTextInput > div > div {{
        background: {colors['bg2']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 25px !important;
    }}
    .stTextInput input {{
        background: transparent !important;
        border: none !important;
        color: {colors['text']} !important;
        font-size: 15px !important;
        padding: 12px 16px !important;
    }}
    .stTextInput input::placeholder {{
        color: {colors['muted']} !important;
    }}
    .stTextInput label {{ display: none !important; }}
    
    .stSelectbox > div > div {{
        background: {colors['bg3']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 18px !important;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        background: {colors['bg3']} !important;
        border: none !important;
    }}
    .stSelectbox [data-baseweb="select"] > div > div {{
        color: {colors['text']} !important;
    }}
    /* Dropdown menu styling */
    [data-baseweb="popover"] {{
        background: {colors['bg2']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 12px !important;
    }}
    [data-baseweb="menu"] {{
        background: {colors['bg2']} !important;
    }}
    [data-baseweb="menu"] li {{
        background: {colors['bg2']} !important;
        color: {colors['text']} !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background: {colors['bg3']} !important;
    }}
    .stSelectbox label {{ display: none !important; }}
    
    /* Buttons - theme aware */
    .stButton > button {{
        background: {colors['bg2']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 12px !important;
        color: {colors['text']} !important;
        font-size: 16px !important;
        padding: 8px 16px !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        background: {colors['accent']} !important;
        color: white !important;
        border-color: {colors['accent']} !important;
    }}
    .stButton > button:active {{
        background: {colors['accent']} !important;
    }}
    
    /* Form submit button */
    .stFormSubmitButton > button {{
        background: {colors['bg3']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 20px !important;
        color: {colors['text']} !important;
    }}
    .stFormSubmitButton > button:hover {{
        background: {colors['accent']} !important;
        color: white !important;
        border-color: {colors['accent']} !important;
    }}
    
    /* File uploader styling - COMPLETE FIX */
    .stFileUploader {{
        max-width: 600px;
        margin: 10px auto;
    }}
    .stFileUploader > div {{
        background: transparent !important;
    }}
    .stFileUploader > div > div {{
        background: transparent !important;
    }}
    .stFileUploader [data-testid="stFileUploaderDropzone"] {{
        background: {colors['bg2']} !important;
        border: 2px dashed {colors['border']} !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }}
    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {colors['accent']} !important;
    }}
    /* All text inside dropzone */
    .stFileUploader [data-testid="stFileUploaderDropzone"] * {{
        color: {colors['text']} !important;
    }}
    .stFileUploader [data-testid="stFileUploaderDropzone"] span {{
        color: {colors['text']} !important;
    }}
    .stFileUploader [data-testid="stFileUploaderDropzone"] p {{
        color: {colors['text']} !important;
    }}
    .stFileUploader [data-testid="stFileUploaderDropzone"] small {{
        color: {colors['text2']} !important;
    }}
    .stFileUploader [data-testid="stFileUploaderDropzone"] svg {{
        fill: {colors['text2']} !important;
        stroke: {colors['text2']} !important;
    }}
    .stFileUploader [data-testid="stFileUploaderDropzone"] button {{
        background: {colors['accent']} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }}
    .stFileUploader label {{
        color: {colors['text']} !important;
        font-size: 14px !important;
    }}
    .stFileUploader > section {{
        background: transparent !important;
        border: none !important;
    }}
    .stFileUploader > section > div {{
        background: transparent !important;
    }}
    
    /* Answer box */
    .answer-box {{
        background: {colors['bg2']};
        border: 1px solid {colors['border']};
        border-radius: 16px;
        padding: 24px;
        color: {colors['text']};
        font-size: 15px;
        line-height: 1.8;
    }}
    
    /* Source cards */
    .source-card {{
        background: {colors['bg3']};
        border: 1px solid {colors['border']};
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.2s;
    }}
    .source-card:hover {{
        border-color: {colors['accent']};
    }}
    .source-title {{
        color: {colors['accent']};
        font-size: 13px;
        font-weight: 500;
        text-decoration: none;
    }}
    .source-domain {{
        color: {colors['muted']};
        font-size: 11px;
    }}
    
    /* Query display */
    .query-box {{
        background: {colors['bg2']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 16px;
        margin: 15px 0;
    }}
    .query-text {{
        color: {colors['text']};
        font-size: 17px;
        font-weight: 500;
    }}
    .query-mode {{
        color: {colors['accent']};
        font-size: 12px;
        margin-top: 6px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 1px solid {colors['border']} !important;
        gap: 0 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {colors['text2']} !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {colors['accent']} !important;
        border-bottom-color: {colors['accent']} !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 1rem !important;
    }}
    
    /* Answer text styling */
    .stTabs [data-testid="stMarkdownContainer"] {{
        color: {colors['text']} !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }}
    
    /* Mode desc text */
    .mode-desc {{
        text-align: center;
        color: {colors['muted']};
        font-size: 12px;
        margin-top: 8px;
    }}
    
    /* Column spacing fix */
    [data-testid="column"] {{ padding: 0 2px !important; }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: {colors['bg3']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 8px !important;
        color: {colors['text']} !important;
    }}
    .streamlit-expanderContent {{
        background: {colors['bg2']} !important;
        border: 1px solid {colors['border']} !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        color: {colors['text']} !important;
    }}
    [data-testid="stExpander"] {{
        background: {colors['bg2']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {colors['text']} !important;
    }}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{
        color: {colors['text']} !important;
    }}
    
    /* Spinner and alerts */
    .stSpinner > div {{
        border-color: {colors['accent']} !important;
    }}
    .stAlert {{
        background: {colors['bg2']} !important;
        color: {colors['text']} !important;
        border: 1px solid {colors['border']} !important;
    }}
    
    /* Caption text */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {colors['text2']} !important;
    }}
    
    /* Divider */
    hr {{
        border-color: {colors['border']} !important;
    }}
    </style>
    """

st.markdown(get_css(), unsafe_allow_html=True)


# =====================================
# HELPER FUNCTIONS
# =====================================
def call_api(query: str, mode: str, extra_data: dict = None):
    """Call backend API based on selected mode."""
    mode_config = MODES.get(mode, MODES["Automatic"])
    endpoint = mode_config["endpoint"]
    
    payload = {
        "message": query,
        "workspace_id": WORKSPACE,
        "mode": mode.lower().replace(" ", "_")
    }
    
    # Add extra data for special modes
    if extra_data:
        payload.update(extra_data)
    
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=180)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {
                "answer": f"Error: Invalid JSON response from server",
                "sources": [],
                "links": [],
                "images": [],
                "followups": []
            }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "links": [],
            "images": [],
            "followups": []
        }


def upload_files(files):
    """Upload files to backend."""
    if not files:
        return False
    
    files_payload = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]
    
    try:
        r = requests.post(
            f"{API_URL}/api/upload_docs",
            data={"workspace_id": WORKSPACE},
            files=files_payload,
            timeout=60
        )
        return r.ok
    except:
        return False


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace('www.', '')
    except:
        return url[:30]


# =====================================
# THEME TOGGLE
# =====================================
col_spacer, col_theme = st.columns([12, 1])
with col_theme:
    theme_icon = "🌙" if st.session_state.theme == "dark" else "☀️"
    if st.button(theme_icon, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()


# =====================================
# HERO - Always show
# =====================================
if st.session_state.current_result:
    # Compact version when showing results
    st.markdown("""
    <div class="hero-compact">
        <div class="logo">perplexity<span>clone</span></div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Full version on home
    st.markdown("""
    <div class="hero">
        <div class="logo">perplexity<span>clone</span></div>
        <div class="tagline">Where knowledge begins</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================
# UNIFIED SEARCH BOX (All elements inside)
# =====================================
st.markdown('<div class="search-wrapper">', unsafe_allow_html=True)

# Single row with everything inside
col1, col2, col3, col4 = st.columns([2, 8, 1, 1])

with col1:
    # Mode selector dropdown
    mode_list = list(MODES.keys())
    current_idx = mode_list.index(st.session_state.mode)
    selected = st.selectbox(
        "mode",
        mode_list,
        index=current_idx,
        format_func=lambda x: f"{MODES[x]['icon']} {x}",
        label_visibility="collapsed",
        key="mode_select"
    )
    if selected != st.session_state.mode:
        st.session_state.mode = selected
        st.rerun()

with col2:
    # Search input
    query = st.text_input(
        "search",
        placeholder="Ask anything...",
        label_visibility="collapsed",
        key="query_input"
    )

with col3:
    # File upload icon button - toggles file picker
    if st.button("📎", key="attach_btn", help="Upload files"):
        st.session_state.show_upload = not st.session_state.show_upload

with col4:
    # Submit button
    submit = st.button("→", key="submit_btn", help="Search")

st.markdown('</div>', unsafe_allow_html=True)

# Mode description
st.markdown(f'<div class="mode-desc">{MODES[st.session_state.mode]["icon"]} {st.session_state.mode}: {MODES[st.session_state.mode]["desc"]}</div>', unsafe_allow_html=True)

# =====================================
# SPECIAL UI FOR PRODUCT MVP MODE
# =====================================
if st.session_state.mode == "Product MVP" and not st.session_state.current_result:
    st.markdown("""
    <div style="text-align: center; padding: 20px; margin: 20px auto; max-width: 700px; 
                background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                border-radius: 16px; color: white;">
        <h3 style="margin: 0; font-size: 24px;">🚀 Product Builder – Idea → MVP Blueprint</h3>
        <p style="margin: 10px 0 0; opacity: 0.9;">🟠 Product Builder Active</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #888; margin: 15px 0;'>Describe your product idea:</p>", unsafe_allow_html=True)

# =====================================
# SPECIAL UI FOR VIDEO BRAIN MODE
# =====================================
if st.session_state.mode == "Video Brain" and not st.session_state.current_result:
    st.markdown("""
    <div style="text-align: center; padding: 20px; margin: 20px auto; max-width: 700px; 
                background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); 
                border-radius: 16px; color: white;">
        <h3 style="margin: 0; font-size: 24px;">🎥 Video Brain – Understand Any YouTube Lecture</h3>
        <p style="margin: 10px 0 0; opacity: 0.9;">🔵 Upload Video First</p>
    </div>
    """, unsafe_allow_html=True)
    
    # YouTube URL input
    col_yt1, col_yt2 = st.columns([5, 1])
    with col_yt1:
        youtube_url = st.text_input(
            "youtube_url",
            placeholder="Enter YouTube URL (e.g., https://youtube.com/watch?v=...)",
            label_visibility="collapsed",
            key="youtube_url_input"
        )
    with col_yt2:
        if st.button("📺 Load", key="load_video_btn"):
            if youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
                st.session_state.youtube_url = youtube_url
                st.session_state.video_loaded = True
                st.success("✅ Video loaded! Now ask questions about it.")
            else:
                st.error("Please enter a valid YouTube URL")
    
    if st.session_state.video_loaded and st.session_state.youtube_url:
        st.success(f"📺 Video ready: {st.session_state.youtube_url[:50]}...")
        st.markdown("<p style='text-align: center; color: #888; margin: 15px 0;'>Ask about the video:</p>", unsafe_allow_html=True)

# Show file uploader when icon is clicked
if st.session_state.show_upload:
    uploaded = st.file_uploader(
        "Upload documents (PDF, TXT, MD, PPTX)",
        type=["pdf", "txt", "md", "pptx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded:
        with st.spinner("📤 Uploading..."):
            if upload_files(uploaded):
                new_files = [f.name for f in uploaded if f.name not in st.session_state.uploaded_files]
                if new_files:
                    st.session_state.uploaded_files.extend(new_files)
                    st.success(f"✅ {len(new_files)} file(s) uploaded!")
                    st.session_state.show_upload = False
                    st.rerun()

# Show uploaded files count
if st.session_state.uploaded_files:
    st.caption(f"📁 {len(st.session_state.uploaded_files)} file(s) ready for RAG")


# =====================================
# HANDLE SEARCH
# =====================================
if submit and query.strip():
    extra_data = None
    
    # For Video Brain mode, include the YouTube URL
    if st.session_state.mode == "Video Brain":
        if st.session_state.video_loaded and st.session_state.youtube_url:
            extra_data = {"youtube_url": st.session_state.youtube_url}
        else:
            st.warning("⚠️ Please load a YouTube video first!")
            st.stop()
    
    # For Product MVP mode, save to ideas history
    if st.session_state.mode == "Product MVP":
        st.session_state.product_ideas.append({
            "idea": query.strip(),
            "time": "just now"
        })
    
    with st.spinner(f"🔄 {st.session_state.mode}..."):
        result = call_api(query.strip(), st.session_state.mode, extra_data)
        st.session_state.current_result = {
            "query": query.strip(),
            "mode": st.session_state.mode,
            "data": result
        }
    st.rerun()


# =====================================
# DISPLAY RESULTS
# =====================================
if st.session_state.current_result:
    result = st.session_state.current_result
    data = result["data"]
    
    st.divider()
    
    # Special header for Product MVP mode
    if result['mode'] == "Product MVP":
        st.markdown("""
        <div style="text-align: center; padding: 15px; margin: 10px auto; max-width: 700px; 
                    background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                    border-radius: 12px; color: white;">
            <h4 style="margin: 0;">📄 MVP Blueprint</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Special header for Video Brain mode
    if result['mode'] == "Video Brain":
        st.markdown("""
        <div style="text-align: center; padding: 15px; margin: 10px auto; max-width: 700px; 
                    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); 
                    border-radius: 12px; color: white;">
            <h4 style="margin: 0;">🎥 Video Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Query box
    mode_info = MODES.get(result['mode'], MODES['Automatic'])
    st.markdown(f"""
    <div class="query-box">
        <div class="query-text">{result['query']}</div>
        <div class="query-mode">{mode_info['icon']} {result['mode']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sources count
    sources = data.get("sources", []) or data.get("links", [])
    if sources:
        st.success(f"✓ {len(sources)} sources")
    
    # Memory saved notification for Product MVP
    if result['mode'] == "Product MVP":
        st.info("📝 New Memory Saved")
    
    # Layout - Full width (removed duplicate sidebar sources)
    tabs = st.tabs(["✨ Answer", "🔗 Sources", "🖼️ Images"])
    
    with tabs[0]:
        answer = data.get("answer", "No answer.")
        
        # Display answer directly with markdown
        st.markdown(answer)
        
        followups = data.get("followups", [])
        if followups:
            st.markdown("**Related:**")
            for i, fu in enumerate(followups[:3]):
                if st.button(f"→ {fu}", key=f"fu_{i}"):
                    extra = None
                    if st.session_state.mode == "Video Brain" and st.session_state.youtube_url:
                        extra = {"youtube_url": st.session_state.youtube_url}
                    with st.spinner("..."):
                        new_result = call_api(fu, st.session_state.mode, extra)
                        st.session_state.current_result = {
                            "query": fu,
                            "mode": st.session_state.mode,
                            "data": new_result
                        }
                    st.rerun()
    
    with tabs[1]:
        links = data.get("links", [])
        if links:
            for link in links:
                st.markdown(f"""
                <div class="source-card">
                    <a href="{link.get('url','#')}" target="_blank" class="source-title">{link.get('title','Source')}</a>
                    <div class="source-domain">{get_domain(link.get('url',''))}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No sources")
    
    with tabs[2]:
        images = data.get("images", [])
        if images:
            cols = st.columns(3)
            for i, img in enumerate(images[:9]):
                url = img.get("url") or img.get("thumbnail_url")
                if url:
                    with cols[i % 3]:
                        st.image(url, use_container_width=True)
        else:
            st.info("No images")


# =====================================
# SIDEBAR (for settings)
# =====================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.current_result = None
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🗑️ Clear Files", use_container_width=True):
        st.session_state.uploaded_files = []
        st.info("Files cleared")
    
    if st.button("🗑️ Clear Video", use_container_width=True):
        st.session_state.youtube_url = ""
        st.session_state.video_loaded = False
        st.info("Video cleared")
    
    st.divider()
    st.caption(f"Theme: {'🌙 Dark' if st.session_state.theme == 'dark' else '☀️ Light'}")
    st.caption(f"Mode: {st.session_state.mode}")
    
    if st.session_state.uploaded_files:
        st.divider()
        st.markdown("### 📁 Files")
        for f in st.session_state.uploaded_files:
            st.caption(f"📄 {f}")
    
    # Show video info for Video Brain mode
    if st.session_state.video_loaded and st.session_state.youtube_url:
        st.divider()
        st.markdown("### 🎥 Loaded Video")
        st.caption(f"📺 {st.session_state.youtube_url[:40]}...")
    
    # Show recent product ideas
    if st.session_state.product_ideas:
        st.divider()
        st.markdown("### 🧾 Recent Ideas")
        for idea in st.session_state.product_ideas[-3:]:
            st.caption(f"💡 {idea['idea'][:30]}...")
