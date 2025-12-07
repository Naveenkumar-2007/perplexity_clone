# ===================== api.py ==========================
from typing import List, Dict
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.config import Config
from config.system_prompt import PPLX_SYSTEM_PROMPT

# Core routing / graph
from rag.router import RouterAgent
from rag.graph_deep import (
    DeepResearchGraph,
    WebSearchGraph,
    RAGOnlyGraph,
    AgenticRAGGraph,
    AnalysisGraph,
    SummarizeGraph
)

# Tools
from tools.memory_tool import MemoryTool
from tools.name_tool import NameTool
from tools.search_tool import SearchTool
from tools.browse_tool import BrowseTool
from tools.reranker_tool import Reranker
from tools.followup_tool import FollowUpGenerator
from tools.image_tavily import TavilyImageSearch
from tools.knowledge_panel import KnowledgePanel
from tools.summarizer_tool import SummarizerTool

# RAG pipeline
from document_processing.processor import DocumentProcessor
from vectorstore.store import VectorStore

# File Manager for per-workspace RAG
from files.file_manager import FileManager


# =======================================================
# FastAPI App
# =======================================================
app = FastAPI(title="Perplexity Clone API", version="8.0 - Production LangGraph")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
    allow_credentials=True,
)


# =======================================================
# Health Check Endpoint (for Azure Container Apps)
# =======================================================
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "perplexity-clone-api"}


# =======================================================
# Global Components
# =======================================================
llm = Config.get_llm()
router = RouterAgent()

memory = MemoryTool()
name_tool = NameTool()
followup = FollowUpGenerator()
search_tool = SearchTool()
browse_tool = BrowseTool()
reranker = Reranker()
image_search = TavilyImageSearch()
knowledge_panel = KnowledgePanel()
summarizer = SummarizerTool()

# RAG demo vectorstore
processor = DocumentProcessor(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
)
demo_docs = processor.load_url("https://lilianweng.github.io/posts/2023-06-23-agent/")
demo_splits = processor.split(demo_docs)

vector = VectorStore()
vector.create(demo_splits)

# File manager for per-workspace document RAG
file_manager = FileManager(base_dir="workspace_data")

# =======================================================
# Initialize All LangGraph Pipelines
# =======================================================
deep_graph = DeepResearchGraph(vector)
web_graph = WebSearchGraph()
rag_graph = RAGOnlyGraph(file_manager)
agentic_graph = AgenticRAGGraph(file_manager, vector, image_search)
analysis_graph = AnalysisGraph()
summarize_graph = SummarizeGraph()

print("✅ All LangGraph pipelines initialized!")


# =======================================================
# Models
# =======================================================
class ChatRequest(BaseModel):
    message: str
    workspace_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]] = []
    links: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []
    followups: List[str] = []
    default_tab: str = "answer"  # "answer" | "links" | "images"
    workspace_id: str


# =======================================================
# Utils
# =======================================================
def build_context(ws: str, new_msg: str):
    """Inject full workspace chat history + system prompt."""
    messages = [{"role": "system", "content": PPLX_SYSTEM_PROMPT}]
    for msg in memory.get_long_chat(ws):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": new_msg})
    return messages


def guess_default_tab(query: str, mode: str) -> str:
    """Decide which UI tab should be first (Answer / Links / Images)."""
    q = query.lower()
    image_words = [
        "image", "images", "photo", "photos", "picture", "pictures",
        "wallpaper", "logo", "flag", "screenshot", "pic"
    ]
    if any(w in q for w in image_words):
        return "images"
    if mode == "web":
        return "links"
    return "answer"


def tavily_images_safe(query: str) -> List[Dict[str, str]]:
    """Safe wrapper around Tavily image search."""
    try:
        return image_search.search(query, count=6)
    except Exception as e:
        print("Tavily image search error:", e)
        return []


def convert_links(results: List[Dict]) -> List[Dict[str, str]]:
    """Convert Tavily web search results to link objects."""
    links = []
    for r in results:
        url = r.get("url")
        if not url:
            continue
        links.append(
            {
                "title": r.get("title", "Result"),
                "url": url,
                "snippet": (r.get("content") or "")[:200],
            }
        )
    return links


# =======================================================
# Chat Endpoint
# =======================================================
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    q = req.message.strip()
    ws = req.workspace_id

    memory.add(ws, "user", q)

    # -------- Name memory special cases --------
    extracted = name_tool.extract_name(q)
    if extracted:
        memory.set_name(ws, extracted)
        reply = f"Nice to meet you, {extracted}! I’ll remember your name."
        memory.add(ws, "assistant", reply)
        return ChatResponse(answer=reply, workspace_id=ws)

    if q.lower() in ["tell me my name", "what is my name"]:
        nm = memory.get_name(ws)
        ans = f"Your name is {nm} 😊" if nm else "You haven’t told me your name yet."
        memory.add(ws, "assistant", ans)
        return ChatResponse(answer=ans, workspace_id=ws)

    # -------- Routing --------
    mode = router.route(q)
    default_tab = guess_default_tab(q, mode)

    answer = ""
    links: List[Dict[str, str]] = []
    sources: List[Dict[str, str]] = []
    follow: List[str] = []

    # -------- LLM Mode (chat / creative / small talk) --------
    if mode == "llm":
        msgs = build_context(ws, q)
        answer = llm.invoke(msgs).content
        follow = followup.generate(answer, q)

        # Optional small set of links
        try:
            res = search_tool.search(q, num_results=3)
            links = convert_links(res)
        except Exception as e:
            print("search error (llm mode):", e)

    # -------- Image Mode (image search queries) --------
    elif mode == "image":
        # For image queries, provide brief context + focus on images tab
        try:
            res = search_tool.search(q, num_results=3)
            ctx = res[0].get("snippet", "") if res else ""
            answer = f"Here are images related to '{q}'."
            if ctx:
                answer += f"\n\n{ctx}"
            links = convert_links(res)
        except Exception as e:
            print("search error (image mode):", e)
            answer = f"Showing images for: {q}"
        
        follow = []

    # -------- AGENTIC RAG Mode (files + web + images + knowledge) --------
    elif mode == "rag":
        # PLANNER AGENT: Decide which agents to activate
        q_lower = q.lower()
        
        use_file_rag = any(
            w in q_lower for w in [
                "summarize", "according to", "in this pdf", "in the document",
                "based on the file", "read my", "extract from", "uploaded",
                "this file", "the file", "my file", "from file"
            ]
        ) or len(q.split()) > 2  # multi-word questions likely need file RAG
        
        use_web = any(
            w in q_lower for w in [
                "today", "latest", "current", "news", "stock", "price",
                "real-time", "weather", "who is", "what is", "where is",
                "when", "how much", "compare"
            ]
        )
        
        use_images = any(
            w in q_lower for w in [
                "image", "images", "logo", "flag", "photos", "look like", 
                "picture", "show me", "wallpaper", "screenshot"
            ]
        )
        
        # FILE AGENT: Retrieve from workspace uploaded docs
        ws_obj = file_manager.get_workspace(ws)
        file_chunks = []
        if use_file_rag and ws_obj.initialized:
            file_chunks = ws_obj.retrieve(q, k=6)
        
        # REFERENCE AGENT: Retrieve from base vector store (demo docs)
        base_chunks = vector.retrieve(q, k=4)
        base_chunks = reranker.rerank(q, base_chunks, top_k=3)
        
        # WEB AGENT: Fetch live web content
        web_pages = []
        web_results = []
        if use_web:
            try:
                web_results = search_tool.search(q, num_results=4)
                for r in web_results:
                    url = r.get("url")
                    if not url:
                        continue
                    text = browse_tool.fetch_clean(url)
                    if text:
                        web_pages.append({
                            "title": r.get("title", ""),
                            "url": url,
                            "content": text[:1500]  # Speed optimization
                        })
            except Exception as e:
                print(f"Web agent error: {e}")
        
        # IMAGE AGENT: Fetch relevant images
        images_result = tavily_images_safe(q) if use_images else []
        
        # BUILD COMBINED CONTEXT
        contexts = []
        
        if file_chunks:
            file_ctx = "\n\n".join(d.page_content for d in file_chunks)
            contexts.append(f"📄 FILE CONTEXT (from uploaded documents):\n{file_ctx}")
        
        if base_chunks:
            ref_ctx = "\n\n".join(d.page_content for d in base_chunks)
            contexts.append(f"📚 REFERENCE CONTEXT:\n{ref_ctx}")
        
        if web_pages:
            web_ctx = "\n\n".join(f"[{p['title']}]: {p['content']}" for p in web_pages)
            contexts.append(f"🌐 WEB CONTEXT (live web data):\n{web_ctx}")
        
        full_context = "\n\n-----\n\n".join(contexts) if contexts else "No context available."
        
        # SYNTHESIZER AGENT: Generate final answer
        synth_prompt = f"""You are an AGENTIC RAG synthesis model like Perplexity AI.
Combine information from FILE CONTEXT, REFERENCE CONTEXT and WEB CONTEXT.

RULES:
1. PRIORITIZE info from FILE CONTEXT (user's uploaded documents) when available.
2. Use WEB CONTEXT to add current/live information.
3. Use REFERENCE CONTEXT for background knowledge.
4. Cite sources using [1], [2], etc. when referencing specific info.
5. If answering from a file, say "According to your uploaded document..."
6. Do NOT hallucinate - only use info from the provided contexts.
7. Be concise but comprehensive.

AVAILABLE CONTEXT:
{full_context}

USER QUESTION: {q}

FINAL ANSWER:"""
        
        msgs = build_context(ws, synth_prompt)
        answer = llm.invoke(msgs).content
        follow = followup.generate(answer, q)
        
        # BUILD SOURCES
        sources = []
        if file_chunks:
            for d in file_chunks:
                sources.append({
                    "title": d.metadata.get("source", "📄 Uploaded File"),
                    "url": d.metadata.get("file_path", "")
                })
        
        if web_pages:
            for p in web_pages:
                sources.append({"title": p["title"], "url": p["url"]})
        
        # BUILD LINKS
        links = convert_links(web_results)
        
        # Set images from image agent
        if images_result:
            # Will be set at the end with tavily_images_safe
            pass

    # -------- Web Mode (real-time / entities / news) --------
    elif mode == "web":
        res = search_tool.search(q, num_results=5)

        pages = []
        for r in res:
            url = r.get("url")
            if not url:
                continue
            text = browse_tool.fetch_clean(url)
            if not text:
                continue
            pages.append(
                {
                    "title": r.get("title", "Webpage"),
                    "url": url,
                    "content": text[:2000],
                }
            )

        ctx = "\n\n".join(p["content"] for p in pages)
        prompt = (
            "Use ONLY the following web content to answer. "
            "Cite sources using [1], [2], etc.\n\n"
            f"{ctx}\n\nQuestion: {q}"
        )

        msgs = build_context(ws, prompt)
        answer = llm.invoke(msgs).content
        follow = followup.generate(answer, q)

        links = [
            {
                "title": p["title"],
                "url": p["url"],
                "snippet": p["content"][:200],
            }
            for p in pages
        ]

        sources = [{"title": p["title"], "url": p["url"]} for p in pages]

    # -------- Fallback → LLM --------
    else:
        msgs = build_context(ws, q)
        answer = llm.invoke(msgs).content
        follow = followup.generate(answer, q)

    # -------- Images (for Images tab) --------
    images = tavily_images_safe(q)
    
    # Debug logging
    print(f"\n=== API Response Debug ===")
    print(f"Mode: {mode}")
    print(f"Links count: {len(links)}")
    print(f"Images count: {len(images)}")
    print(f"Sources count: {len(sources)}")
    if links:
        print(f"First link: {links[0]}")
    if images:
        print(f"First image: {images[0]}")
    print(f"========================\n")

    memory.add(ws, "assistant", answer)

    return ChatResponse(
        answer=answer,
        sources=sources,
        links=links,
        images=images,
        followups=follow,
        default_tab=default_tab,
        workspace_id=ws,
    )


# =======================================================
# Streaming Endpoint
# =======================================================
@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):

    q = req.message
    ws = req.workspace_id
    memory.add(ws, "user", q)

    msgs = build_context(ws, q)

    def generate():
        full = ""
        for chunk in llm.stream(msgs):
            tok = getattr(chunk, "content", "")
            if tok:
                full += tok
                yield tok
        memory.add(ws, "assistant", full)

    return StreamingResponse(generate(), media_type="text/plain")


# =======================================================
# Deep Research Endpoint
# =======================================================
@app.post("/api/deep_research", response_model=ChatResponse)
def deep_research(req: ChatRequest):

    q = req.message
    ws = req.workspace_id

    memory.add(ws, "user", q)

    try:
        state = deep_graph.run(q)
        # state is a dict (TypedDict), not an object
        answer = state.get("final_answer", "No answer generated.")
        sources = state.get("sources", [])
    except Exception as e:
        print("Deep research error:", e)
        answer = "Something went wrong in deep research mode."
        sources = []

    memory.add(ws, "assistant", answer)
    images = tavily_images_safe(q)
    follow = followup.generate(answer, q)

    return ChatResponse(
        answer=answer,
        sources=sources,
        links=[],
        images=images,
        followups=follow,
        default_tab="answer",
        workspace_id=ws,
    )


# =======================================================
# Knowledge Panel Endpoint
# =======================================================
@app.get("/api/knowledge_panel")
def get_knowledge_panel(q: str):
    """
    Returns Wikipedia-style infobox + AI-generated facts.
    Used by UI to render a sidebar knowledge card.
    """
    try:
        panel = knowledge_panel.build_panel(q)
        return panel
    except Exception as e:
        print("Knowledge panel error:", e)
        return {"wiki": {}, "facts": []}


# =======================================================
# FILE UPLOAD (PDF / TXT / PPTX) - Perplexity Spaces Feature
# =======================================================
@app.post("/api/upload_docs")
async def upload_docs(
    workspace_id: str = Form("default"),
    files: List[UploadFile] = File(...)
):
    """
    Upload one or more documents and index them for this workspace.
    Supports PDF, TXT, MD, PPT, PPTX files.
    """
    ws = file_manager.get_workspace(workspace_id)
    saved_paths = []

    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in [".pdf", ".txt", ".md", ".ppt", ".pptx"]:
            continue  # skip unsupported types

        dest = Path(ws.base_dir) / f.filename
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_paths.append(dest)

    if saved_paths:
        ws.add_files(saved_paths)
        print(f"✅ Indexed {len(saved_paths)} files for workspace '{workspace_id}'")

    return {
        "workspace_id": workspace_id,
        "files": ws.files,
        "count": len(ws.files),
        "message": f"Successfully indexed {len(saved_paths)} files"
    }


@app.get("/api/workspace_files/{workspace_id}")
def get_workspace_files(workspace_id: str):
    """Get list of files uploaded to a workspace."""
    ws = file_manager.get_workspace(workspace_id)
    return {
        "workspace_id": workspace_id,
        "files": ws.files,
        "initialized": ws.initialized
    }


@app.delete("/api/workspace/{workspace_id}")
def clear_workspace(workspace_id: str):
    """Clear all files from a workspace."""
    file_manager.clear_workspace(workspace_id)
    return {"message": f"Workspace '{workspace_id}' cleared"}


# =======================================================
# MODE-SPECIFIC ENDPOINTS
# =======================================================

class ModeRequest(BaseModel):
    message: str
    workspace_id: str = "default"
    mode: str = "auto"


@app.post("/api/focus", response_model=ChatResponse)
def focus_mode(req: ModeRequest):
    """Focus mode - concise, direct answers without web search."""
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    prompt = f"""You are in FOCUS mode. Provide a concise, direct answer.
- No unnecessary elaboration
- Get straight to the point
- Use bullet points if helpful
- Be accurate and helpful

Question: {q}

Answer:"""
    
    msgs = build_context(ws, prompt)
    answer = llm.invoke(msgs).content
    follow = followup.generate(answer, q)
    
    memory.add(ws, "assistant", answer)
    
    return ChatResponse(
        answer=answer,
        sources=[],
        links=[],
        images=[],
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


@app.post("/api/writing", response_model=ChatResponse)
def writing_mode(req: ModeRequest):
    """Writing mode - creative writing, essays, content generation."""
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    prompt = f"""You are in WRITING mode - a creative writing assistant.
Help with:
- Essays, articles, blog posts
- Creative writing, stories
- Professional emails and documents
- Content improvement and editing
- Grammar and style suggestions

Be creative, engaging, and helpful. Format your response well.

Request: {q}

Response:"""
    
    msgs = build_context(ws, prompt)
    answer = llm.invoke(msgs).content
    follow = followup.generate(answer, q)
    
    memory.add(ws, "assistant", answer)
    
    return ChatResponse(
        answer=answer,
        sources=[],
        links=[],
        images=[],
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


@app.post("/api/math", response_model=ChatResponse)
def math_mode(req: ModeRequest):
    """Math mode - mathematical calculations and explanations."""
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    prompt = f"""You are in MATH mode - a mathematical assistant.
- Solve mathematical problems step by step
- Show all work and calculations
- Explain the reasoning
- Use proper mathematical notation
- Handle algebra, calculus, statistics, geometry, etc.

Problem: {q}

Solution:"""
    
    msgs = build_context(ws, prompt)
    answer = llm.invoke(msgs).content
    follow = followup.generate(answer, q)
    
    memory.add(ws, "assistant", answer)
    
    return ChatResponse(
        answer=answer,
        sources=[],
        links=[],
        images=[],
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


@app.post("/api/code", response_model=ChatResponse)
def code_mode(req: ModeRequest):
    """Code mode - programming help and code generation."""
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    prompt = f"""You are in CODE mode - an expert programming assistant.
- Write clean, efficient, well-commented code
- Explain the code logic
- Follow best practices
- Handle any programming language
- Debug and fix code issues
- Suggest improvements

Request: {q}

Response:"""
    
    msgs = build_context(ws, prompt)
    answer = llm.invoke(msgs).content
    follow = followup.generate(answer, q)
    
    memory.add(ws, "assistant", answer)
    
    return ChatResponse(
        answer=answer,
        sources=[],
        links=[],
        images=[],
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


@app.post("/api/analyze", response_model=ChatResponse)
def analyze_mode(req: ModeRequest):
    """
    Analysis mode - deep analysis with web research.
    Production-level LangGraph implementation.
    """
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    # Run the AnalysisGraph pipeline
    state = analysis_graph.run(q)
    
    answer = state.get("answer", "No analysis generated.")
    sources = state.get("sources", [])
    links = state.get("links", [])
    follow = state.get("followups", [])
    
    # Get related images
    images = tavily_images_safe(q)
    
    memory.add(ws, "assistant", answer)
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        links=links,
        images=images,
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


@app.post("/api/summarize", response_model=ChatResponse)
def summarize_mode(req: ModeRequest):
    """
    Summarize mode - summarize uploaded documents OR web content.
    Prioritizes uploaded files, then falls back to web search.
    """
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    # STEP 1: Check for uploaded files first
    ws_obj = file_manager.get_workspace(ws)
    
    if ws_obj.initialized and ws_obj.files:
        # Summarize from uploaded files
        print(f"📝 SUMMARIZE MODE: Using uploaded files")
        try:
            # Retrieve relevant chunks from files
            chunks = ws_obj.retrieve(q, k=10)
            if chunks:
                # Combine chunk content for summarization
                content = "\n\n".join([c.page_content for c in chunks])
                
                # Generate summary
                summary = summarizer.summarize(content, max_words=400)
                
                # Build sources from files
                seen_files = set()
                sources = []
                for c in chunks:
                    fname = c.metadata.get("source", "Document")
                    if fname not in seen_files:
                        sources.append({"title": f"📄 {fname}", "url": ""})
                        seen_files.add(fname)
                
                follow = followup.generate(summary, q)
                
                memory.add(ws, "assistant", summary)
                
                return ChatResponse(
                    answer=summary,
                    sources=sources,
                    links=[],
                    images=[],
                    followups=follow,
                    default_tab="answer",
                    workspace_id=ws
                )
        except Exception as e:
            print(f"  ❌ File summarize error: {e}")
    
    # STEP 2: Check if it's a URL
    if q.startswith("http"):
        print(f"📝 SUMMARIZE MODE: URL detected")
        try:
            content = browse_tool.fetch_clean(q)
            if content:
                summary = summarizer.summarize(content, max_words=400)
                sources = [{"title": "Source URL", "url": q}]
                links = [{"title": "Source", "url": q, "snippet": content[:200]}]
                follow = followup.generate(summary, q)
                
                memory.add(ws, "assistant", summary)
                
                return ChatResponse(
                    answer=summary,
                    sources=sources,
                    links=links,
                    images=[],
                    followups=follow,
                    default_tab="answer",
                    workspace_id=ws
                )
        except Exception as e:
            print(f"  ❌ URL fetch error: {e}")
    
    # STEP 3: Fall back to web search and summarize
    print(f"📝 SUMMARIZE MODE: Web search fallback")
    try:
        results = search_tool.search(q, num_results=3)
        content_parts = []
        links = []
        
        for r in results:
            url = r.get("url", "")
            title = r.get("title", "")
            text = browse_tool.fetch_clean(url)
            if text:
                content_parts.append(text[:1500])
                links.append({"title": title, "url": url, "snippet": text[:150]})
        
        if content_parts:
            combined = "\n\n".join(content_parts)
            summary = summarizer.summarize(combined, max_words=400)
        else:
            summary = "Could not find content to summarize."
        
        sources = [{"title": l["title"], "url": l["url"]} for l in links]
        follow = followup.generate(summary, q)
        
        memory.add(ws, "assistant", summary)
        
        return ChatResponse(
            answer=summary,
            sources=sources,
            links=links,
            images=[],
            followups=follow,
            default_tab="answer",
            workspace_id=ws
        )
    except Exception as e:
        print(f"  ❌ Summarize error: {e}")
        return ChatResponse(
            answer=f"Error generating summary: {str(e)}",
            sources=[],
            links=[],
            images=[],
            followups=[],
            default_tab="answer",
            workspace_id=ws
        )


# =======================================================
# PRODUCTION-LEVEL MODE ENDPOINTS
# =======================================================

@app.post("/api/web", response_model=ChatResponse)
def web_search_mode(req: ModeRequest):
    """
    Web Search Mode - Real-time web search with source citations.
    Production-level LangGraph implementation.
    """
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    # Run the WebSearchGraph pipeline
    state = web_graph.run(q)
    
    answer = state.get("answer", "No answer generated.")
    sources = state.get("sources", [])
    links = state.get("links", [])
    follow = state.get("followups", [])
    
    # Get images separately
    images = tavily_images_safe(q)
    
    memory.add(ws, "assistant", answer)
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        links=links,
        images=images,
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


@app.post("/api/rag", response_model=ChatResponse)
def rag_mode(req: ModeRequest):
    """
    RAG Mode - Search uploaded documents only.
    Production-level LangGraph implementation.
    """
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    
    # Run the RAGOnlyGraph pipeline
    state = rag_graph.run(q, ws)
    
    answer = state.get("answer", "No answer generated.")
    sources = state.get("sources", [])
    follow = state.get("followups", [])
    
    memory.add(ws, "assistant", answer)
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        links=[],
        images=[],
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


@app.post("/api/agentic", response_model=ChatResponse)
def agentic_mode(req: ModeRequest):
    """
    Agentic Mode - Multi-agent RAG with Planner, File, Web, Knowledge, Image, Synthesizer.
    Production-level LangGraph implementation.
    """
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    print(f"\n🤖 AGENTIC MODE (LangGraph): {q}")
    
    # Run the AgenticRAGGraph pipeline
    state = agentic_graph.run(q, ws)
    
    answer = state.get("answer", "No answer generated.")
    sources = state.get("sources", [])
    links = state.get("links", [])
    images = state.get("images", [])
    follow = state.get("followups", [])
    
    memory.add(ws, "assistant", answer)
    print(f"  ✅ AgenticGraph: Completed with {len(sources)} sources")
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        links=links,
        images=images,
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


# =======================================================
# PRODUCT MVP ENDPOINT - Generates MVP Blueprints
# =======================================================
class ProductMVPRequest(BaseModel):
    message: str
    workspace_id: str = "default"
    mode: str = "product_mvp"


@app.post("/api/product_mvp", response_model=ChatResponse)
def product_mvp_mode(req: ProductMVPRequest):
    """
    Product MVP Mode - Generates comprehensive MVP blueprints from product ideas.
    Includes product name, pitch, target users, features, architecture, tech stack, and more.
    """
    q = req.message.strip()
    ws = req.workspace_id
    
    memory.add(ws, "user", q)
    print(f"\n🚀 PRODUCT MVP MODE: {q}")
    
    # Research similar products and market
    market_research = ""
    try:
        results = search_tool.search(f"{q} startup MVP product", num_results=3)
        if results:
            for r in results:
                url = r.get("url", "")
                text = browse_tool.fetch_clean(url)
                if text:
                    market_research += text[:800] + "\n\n"
    except Exception as e:
        print(f"Market research error: {e}")
    
    prompt = f"""You are a PRODUCT BUILDER AI that creates comprehensive MVP blueprints.

The user wants to build: {q}

{f"MARKET RESEARCH (use for context):{chr(10)}{market_research}" if market_research else ""}

Generate a COMPLETE MVP Blueprint with the following sections. Use markdown formatting with tables where appropriate:

# 📄 MVP Blueprint – [Product Name]
A one-line description of the product.

## 1. Product Name
Create a catchy, memorable product name.

## 2. One‑Line Pitch
A compelling pitch in quotes that explains the value proposition.

## 3. Target Users
Create a markdown table with columns: Persona | Age | Occupation | Goals | Pain Points | How [Product] Helps
Include 4-5 different user personas.

## 4. Problems to Solve
List 5 key problems the product solves with bullet points.

## 5. MVP Features
Create a table with: Feature | Description | Priority (Must-have/Nice-to-have)
Include 8-10 features.

## 6. User Journey (Step‑by‑Step)
Number each step of the user journey from landing to retention.

## 7. System Architecture
Create an ASCII diagram showing the system components and their connections.
Include: Frontend, Backend, Database, APIs, Third-party services.

## 8. Database Tables
Create a table showing the main database tables with: Table | Columns | Notes

## 9. API Endpoints (REST)
Create a table with: Method | Endpoint | Description | Auth Required

## 10. Tech Stack
Create a table with: Layer | Technology | Reason
Cover: Frontend, Backend, Auth, Database, Cache, Storage, Hosting, CI/CD, Monitoring

## 11. Future Features (Post‑MVP)
List 8 features for after MVP launch.

## Next Steps
List 5 actionable next steps to start building.

End with: **Happy building! 🚀**

Be detailed, practical, and use real-world best practices. Make it production-ready."""

    msgs = build_context(ws, prompt)
    answer = llm.invoke(msgs).content
    
    # Generate follow-up questions
    follow = [
        "Generate wireframes for core screens",
        "Create a development timeline",
        "Estimate the MVP budget",
        "Design the database schema in detail",
        "Write user stories for MVP features"
    ]
    
    memory.add(ws, "assistant", answer)
    print(f"  ✅ Product MVP: Blueprint generated")
    
    return ChatResponse(
        answer=answer,
        sources=[],
        links=[],
        images=[],
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )


# =======================================================
# VIDEO BRAIN ENDPOINT - YouTube Video Analysis
# =======================================================
class VideoBrainRequest(BaseModel):
    message: str
    workspace_id: str = "default"
    mode: str = "video_brain"
    youtube_url: str = ""


@app.post("/api/video_brain", response_model=ChatResponse)
def video_brain_mode(req: VideoBrainRequest):
    """
    Video Brain Mode - Analyzes YouTube videos and answers questions about them.
    Extracts transcript/content and provides intelligent responses.
    """
    q = req.message.strip()
    ws = req.workspace_id
    youtube_url = req.youtube_url
    
    memory.add(ws, "user", q)
    print(f"\n🎥 VIDEO BRAIN MODE: {q}")
    print(f"  📺 YouTube URL: {youtube_url}")
    
    if not youtube_url:
        return ChatResponse(
            answer="⚠️ Please provide a YouTube URL first. Enter the URL in the Video Brain interface and click 'Load' before asking questions.",
            sources=[],
            links=[],
            images=[],
            followups=[],
            default_tab="answer",
            workspace_id=ws
        )
    
    # Try to get video information
    video_content = ""
    video_title = ""
    
    try:
        # Extract video ID
        video_id = ""
        if "v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        
        print(f"  🔍 Video ID: {video_id}")
        
        # Search for video information and related content
        if video_id:
            # Search for the video title and description
            topic_results = search_tool.search(f"youtube {video_id}", num_results=3)
            if topic_results:
                for r in topic_results:
                    title = r.get("title", "")
                    if title and not video_title:
                        video_title = title
                    snippet = r.get("content", "") or r.get("snippet", "")
                    if snippet:
                        video_content += snippet + "\n"
        
        # Search for transcript or summary
        search_query = f"youtube video transcript summary {video_title or video_id}"
        results = search_tool.search(search_query, num_results=3)
        
        for r in results[:2]:
            url = r.get("url", "")
            if url and "youtube.com" not in url:  # Skip YouTube pages, get transcripts
                text = browse_tool.fetch_clean(url)
                if text:
                    video_content += text[:2000] + "\n\n"
        
        print(f"  📝 Content gathered: {len(video_content)} chars")
        
    except Exception as e:
        print(f"  ❌ Video content fetch error: {e}")
    
    prompt = f"""You are VIDEO BRAIN AI - an expert at analyzing and explaining YouTube video content.

VIDEO URL: {youtube_url}
{f"VIDEO TITLE: {video_title}" if video_title else ""}

{f"AVAILABLE VIDEO CONTEXT:{chr(10)}{video_content[:4000]}" if video_content else "Note: Could not fetch video transcript directly. I will provide helpful guidance based on the question and general knowledge."}

USER QUESTION: {q}

Instructions:
1. If context is available, answer based on the video content
2. If the question is about summarizing, provide key points and takeaways
3. If asking about specific topics, explain them clearly
4. Use timestamps if available (e.g., "At around 5:30...")
5. If limited information is available, be honest but still provide helpful guidance
6. Format your response with headers and bullet points for clarity
7. Make the response educational and easy to understand

Provide a comprehensive, helpful response:"""

    msgs = build_context(ws, prompt)
    answer = llm.invoke(msgs).content
    
    # Generate follow-up questions about the video
    follow = [
        "Summarize the main points of this video",
        "What are the key takeaways?",
        "Explain the most important concept covered",
        "What questions should I ask about this topic?",
        "Create study notes from this video"
    ]
    
    sources = [{"title": f"🎥 {video_title or 'YouTube Video'}", "url": youtube_url}]
    links = [{"title": video_title or "YouTube Video", "url": youtube_url, "snippet": "Source video"}]
    
    memory.add(ws, "assistant", answer)
    print(f"  ✅ Video Brain: Response generated")
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        links=links,
        images=[],
        followups=follow,
        default_tab="answer",
        workspace_id=ws
    )