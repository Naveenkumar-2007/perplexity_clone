"""
Production-Level Agents for Perplexity Clone
=============================================
Each agent is a node in LangGraph pipelines.
Agents handle specific tasks and pass state to next nodes.
"""

from typing import List, Dict, Any
from rag.rag_state import (
    RAGState, 
    WebSearchState, 
    RAGOnlyState, 
    AgenticState,
    AnalysisState,
    SummarizeState
)
from config.config import Config
from config.system_prompt import PPLX_SYSTEM_PROMPT
from vectorstore.store import VectorStore
from tools.search_tool import SearchTool
from tools.browse_tool import BrowseTool
from tools.reranker_tool import Reranker
from tools.citation_tool import CitationTool
from tools.summarizer_tool import SummarizerTool
from tools.followup_tool import FollowUpGenerator


# =============================================================================
# DEEP RESEARCH AGENTS (Original)
# =============================================================================

class PlannerAgent:
    """Decomposes question into sub-questions."""

    def __init__(self) -> None:
        self.llm = Config.get_llm()

    def plan(self, state: RAGState) -> RAGState:
        prompt = (
            "Break the following question into 3-5 clear sub-questions.\n"
            "Return them as a numbered list.\n\n"
            f"{state['question']}"
        )
        resp = self.llm.invoke([
            {"role": "system", "content": PPLX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        lines = [l.strip("-• ").strip() for l in resp.content.splitlines() if l.strip()]
        subqs: List[str] = []
        for l in lines:
            if l[0].isdigit() or len(lines) <= 5:
                subqs.append(l)
        state["sub_questions"] = subqs[:5]
        return state


class ResearchAgent:
    """Collects evidence from local RAG + web search."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store
        self.search_tool = SearchTool()
        self.browse_tool = BrowseTool()
        self.reranker = Reranker()

    def research(self, state: RAGState) -> RAGState:
        evidence: List[str] = []
        pages_all: List[Dict] = []

        for sq in state.get("sub_questions", []):
            # Local RAG
            docs = self.vs.retrieve(sq, k=8)
            docs = self.reranker.rerank(sq, docs, top_k=4)
            evidence.extend(d.page_content for d in docs)

            # Web search + browse
            results = self.search_tool.search(sq, num_results=3)
            for r in results:
                url = r.get("url")
                title = r.get("title", "Web result")
                if not url:
                    continue
                content = self.browse_tool.fetch_clean(url)
                if not content:
                    continue
                pages_all.append({"title": title, "url": url, "content": content})
                evidence.append(content[:1500])

        state["web_pages"] = pages_all
        state["evidence"] = evidence
        return state


class AggregatorAgent:
    """Writes draft answers per sub-question from evidence."""

    def __init__(self) -> None:
        self.llm = Config.get_llm()

    def aggregate(self, state: RAGState) -> RAGState:
        drafts: List[str] = []
        for sq in state.get("sub_questions", []):
            context = "\n\n".join(state.get("evidence", [])[:12])
            prompt = (
                "Using the evidence below, answer the sub-question briefly and clearly.\n\n"
                f"Evidence:\n{context}\n\nSub-question: {sq}"
            )
            resp = self.llm.invoke([
                {"role": "system", "content": PPLX_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            drafts.append(f"Sub-question: {sq}\n{resp.content}")

        state["draft_answers"] = drafts
        return state


class WriterAgent:
    """Writes final structured deep-research report."""

    def __init__(self) -> None:
        self.llm = Config.get_llm()

    def write(self, state: RAGState) -> RAGState:
        findings = "\n\n".join(state.get("draft_answers", []))
        prompt = (
            "You are Perplexity in deep research mode.\n"
            "Write a structured answer with sections: Overview, Key Points, Details, Conclusion.\n"
            "Use inline citations like [1], [2] where appropriate.\n\n"
            f"Original question:\n{state['question']}\n\nFindings:\n{findings}"
        )
        resp = self.llm.invoke([
            {"role": "system", "content": PPLX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        state["final_answer"] = resp.content
        return state


class ValidatorAgent:
    """Builds source list & (optionally) validates citations."""

    def __init__(self) -> None:
        self.citation_tool = CitationTool()

    def validate_and_attach(self, state: RAGState) -> RAGState:
        sources: List[Dict] = []
        for p in state.get("web_pages", [])[:10]:
            sources.append({"title": p["title"], "url": p["url"]})

        used_sources = self.citation_tool.attach_sources(state.get("final_answer", ""), sources)

        state["sources"] = used_sources
        return state


# =============================================================================
# WEB SEARCH AGENTS
# =============================================================================

class WebSearchNode:
    """Node 1: Execute web search query."""
    
    def __init__(self):
        self.search_tool = SearchTool()
    
    def search(self, state: WebSearchState) -> WebSearchState:
        query = state.get("query", "")
        print(f"  🔍 WebSearchNode: Searching for '{query[:50]}...'")
        
        try:
            results = self.search_tool.search(query, num_results=6)
            state["search_results"] = results
        except Exception as e:
            print(f"  ❌ WebSearchNode error: {e}")
            state["search_results"] = []
        
        return state


class WebFetchNode:
    """Node 2: Fetch and parse web pages."""
    
    def __init__(self):
        self.browse_tool = BrowseTool()
    
    def fetch(self, state: WebSearchState) -> WebSearchState:
        pages = []
        links = []
        
        for r in state.get("search_results", []):
            url = r.get("url")
            if not url:
                continue
            
            try:
                content = self.browse_tool.fetch_clean(url)
                if content:
                    pages.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "content": content[:2500]
                    })
                    links.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "snippet": content[:200]
                    })
            except:
                continue
        
        print(f"  📄 WebFetchNode: Fetched {len(pages)} pages")
        state["web_pages"] = pages
        state["links"] = links
        return state


class WebContextNode:
    """Node 3: Build context from fetched pages."""
    
    def build_context(self, state: WebSearchState) -> WebSearchState:
        pages = state.get("web_pages", [])
        
        if pages:
            context_parts = []
            for i, p in enumerate(pages):
                context_parts.append(f"[{i+1}] {p['title']}:\n{p['content']}")
            state["context"] = "\n\n---\n\n".join(context_parts)
        else:
            state["context"] = ""
        
        print(f"  📝 WebContextNode: Built context from {len(pages)} sources")
        return state


class WebAnswerNode:
    """Node 4: Generate answer from context."""
    
    def __init__(self):
        self.llm = Config.get_llm()
        self.followup = FollowUpGenerator()
    
    def answer(self, state: WebSearchState) -> WebSearchState:
        query = state.get("query", "")
        context = state.get("context", "")
        
        if context:
            prompt = f"""You are a web search assistant like Perplexity AI.
Use ONLY the following web sources to answer. Cite sources using [1], [2], etc.

WEB SOURCES:
{context}

QUESTION: {query}

Provide a comprehensive, well-cited answer:"""
        else:
            prompt = f"Answer this question: {query}"
        
        resp = self.llm.invoke([
            {"role": "system", "content": PPLX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        answer = resp.content
        state["answer"] = answer
        state["followups"] = self.followup.generate(answer, query)
        
        # Build sources
        sources = [{"title": p["title"], "url": p["url"]} for p in state.get("web_pages", [])]
        state["sources"] = sources
        
        print(f"  ✅ WebAnswerNode: Generated answer")
        return state


# =============================================================================
# RAG-ONLY AGENTS
# =============================================================================

class RAGRetrieveNode:
    """Node 1: Retrieve from uploaded documents."""
    
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.reranker = Reranker()
    
    def retrieve(self, state: RAGOnlyState) -> RAGOnlyState:
        query = state.get("query", "")
        ws_id = state.get("workspace_id", "default")
        
        ws = self.file_manager.get_workspace(ws_id)
        
        if not ws.initialized or not ws.files:
            state["file_chunks"] = []
            print(f"  📁 RAGRetrieveNode: No files in workspace")
            return state
        
        try:
            chunks = ws.retrieve(query, k=8)
            # Convert to dicts for state
            state["file_chunks"] = [
                {"content": c.page_content, "source": c.metadata.get("source", "Document")}
                for c in chunks
            ]
            print(f"  📁 RAGRetrieveNode: Retrieved {len(chunks)} chunks")
        except Exception as e:
            print(f"  ❌ RAGRetrieveNode error: {e}")
            state["file_chunks"] = []
        
        return state


class RAGContextNode:
    """Node 2: Build context from retrieved chunks."""
    
    def build_context(self, state: RAGOnlyState) -> RAGOnlyState:
        chunks = state.get("file_chunks", [])
        
        if chunks:
            context_parts = []
            for i, c in enumerate(chunks):
                context_parts.append(f"[DOC {i+1}] {c['source']}:\n{c['content']}")
            state["context"] = "\n\n---\n\n".join(context_parts)
        else:
            state["context"] = ""
        
        print(f"  📝 RAGContextNode: Built context from {len(chunks)} chunks")
        return state


class RAGAnswerNode:
    """Node 3: Generate answer from document context."""
    
    def __init__(self):
        self.llm = Config.get_llm()
        self.followup = FollowUpGenerator()
    
    def answer(self, state: RAGOnlyState) -> RAGOnlyState:
        query = state.get("query", "")
        context = state.get("context", "")
        chunks = state.get("file_chunks", [])
        
        if not context:
            state["answer"] = "📚 No documents found. Please upload files first using the 📎 button."
            state["sources"] = []
            state["followups"] = []
            return state
        
        prompt = f"""You are a document analysis assistant.
Answer ONLY based on the provided documents. Do NOT use external knowledge.

DOCUMENTS:
{context}

QUESTION: {query}

Instructions:
- Answer based ONLY on document content
- Say "According to your documents..." when citing
- Quote relevant parts when helpful
- If info is not in documents, say so

ANSWER:"""
        
        resp = self.llm.invoke([
            {"role": "system", "content": PPLX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        answer = resp.content
        state["answer"] = answer
        state["followups"] = self.followup.generate(answer, query)
        
        # Build sources from chunks
        seen = set()
        sources = []
        for c in chunks:
            src = c.get("source", "Document")
            if src not in seen:
                sources.append({"title": f"📄 {src}", "url": ""})
                seen.add(src)
        state["sources"] = sources
        
        print(f"  ✅ RAGAnswerNode: Generated answer from {len(sources)} sources")
        return state


# =============================================================================
# AGENTIC RAG AGENTS (Multi-Agent Pipeline)
# =============================================================================

class AgenticPlannerNode:
    """Node 1: Planner agent decides which sub-agents to activate."""
    
    def plan(self, state: AgenticState) -> AgenticState:
        query = state.get("query", "").lower()
        
        # Determine which agents to use
        state["use_file"] = any(w in query for w in [
            "document", "file", "pdf", "uploaded", "summarize my",
            "according to", "in the file", "extract", "my notes"
        ])
        
        state["use_web"] = any(w in query for w in [
            "today", "current", "latest", "news", "weather", "stock",
            "who is", "what is", "where", "when", "price", "live",
            "recent", "update"
        ]) or len(query.split()) <= 4
        
        state["use_images"] = any(w in query for w in [
            "image", "photo", "picture", "logo", "show me", "look like",
            "flag", "screenshot"
        ])
        
        state["use_knowledge"] = any(w in query for w in [
            "explain", "define", "concept", "theory", "how does",
            "what is", "meaning of"
        ])
        
        print(f"  📋 AgenticPlannerNode: file={state['use_file']}, web={state['use_web']}, images={state['use_images']}")
        return state


class AgenticFileNode:
    """Node 2: File agent retrieves from uploaded documents."""
    
    def __init__(self, file_manager):
        self.file_manager = file_manager
    
    def retrieve(self, state: AgenticState) -> AgenticState:
        if not state.get("use_file", False):
            state["file_context"] = ""
            state["file_sources"] = []
            return state
        
        query = state.get("query", "")
        ws_id = state.get("workspace_id", "default")
        ws = self.file_manager.get_workspace(ws_id)
        
        if not ws.initialized:
            state["file_context"] = ""
            state["file_sources"] = []
            return state
        
        try:
            chunks = ws.retrieve(query, k=6)
            if chunks:
                state["file_context"] = "\n\n".join([c.page_content for c in chunks])
                state["file_sources"] = [
                    {"title": f"📄 {c.metadata.get('source', 'Document')}", "url": ""}
                    for c in chunks
                ]
                print(f"  📁 AgenticFileNode: Found {len(chunks)} chunks")
            else:
                state["file_context"] = ""
                state["file_sources"] = []
        except Exception as e:
            print(f"  ❌ AgenticFileNode error: {e}")
            state["file_context"] = ""
            state["file_sources"] = []
        
        return state


class AgenticWebNode:
    """Node 3: Web agent fetches real-time information."""
    
    def __init__(self):
        self.search_tool = SearchTool()
        self.browse_tool = BrowseTool()
    
    def search(self, state: AgenticState) -> AgenticState:
        if not state.get("use_web", False):
            state["web_context"] = ""
            state["web_sources"] = []
            state["links"] = []
            return state
        
        query = state.get("query", "")
        
        try:
            results = self.search_tool.search(query, num_results=4)
            web_parts = []
            sources = []
            links = []
            
            for r in results:
                url = r.get("url")
                title = r.get("title", "")
                if not url:
                    continue
                
                content = self.browse_tool.fetch_clean(url)
                if content:
                    web_parts.append(f"[{title}]: {content[:1500]}")
                    sources.append({"title": title, "url": url})
                    links.append({"title": title, "url": url, "snippet": content[:150]})
            
            state["web_context"] = "\n\n".join(web_parts)
            state["web_sources"] = sources
            state["links"] = links
            print(f"  🌐 AgenticWebNode: Found {len(sources)} sources")
            
        except Exception as e:
            print(f"  ❌ AgenticWebNode error: {e}")
            state["web_context"] = ""
            state["web_sources"] = []
            state["links"] = []
        
        return state


class AgenticKnowledgeNode:
    """Node 4: Knowledge agent retrieves from base vector store."""
    
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.reranker = Reranker()
    
    def retrieve(self, state: AgenticState) -> AgenticState:
        if not state.get("use_knowledge", False):
            state["knowledge_context"] = ""
            return state
        
        query = state.get("query", "")
        
        try:
            chunks = self.vs.retrieve(query, k=4)
            chunks = self.reranker.rerank(query, chunks, top_k=3)
            
            if chunks:
                state["knowledge_context"] = "\n\n".join([c.page_content for c in chunks])
                print(f"  📚 AgenticKnowledgeNode: Found {len(chunks)} chunks")
            else:
                state["knowledge_context"] = ""
                
        except Exception as e:
            print(f"  ❌ AgenticKnowledgeNode error: {e}")
            state["knowledge_context"] = ""
        
        return state


class AgenticImageNode:
    """Node 5: Image agent fetches relevant images."""
    
    def __init__(self, image_search):
        self.image_search = image_search
    
    def search(self, state: AgenticState) -> AgenticState:
        if not state.get("use_images", False):
            state["images"] = []
            return state
        
        query = state.get("query", "")
        
        try:
            images = self.image_search.search(query, count=6)
            state["images"] = images
            print(f"  🖼️ AgenticImageNode: Found {len(images)} images")
        except Exception as e:
            print(f"  ❌ AgenticImageNode error: {e}")
            state["images"] = []
        
        return state


class AgenticSynthesizerNode:
    """Node 6: Synthesizer agent combines all contexts and generates final answer."""
    
    def __init__(self):
        self.llm = Config.get_llm()
        self.followup = FollowUpGenerator()
    
    def synthesize(self, state: AgenticState) -> AgenticState:
        query = state.get("query", "")
        
        # Build combined context
        contexts = []
        if state.get("file_context"):
            contexts.append(f"📄 FROM YOUR DOCUMENTS:\n{state['file_context'][:2500]}")
        if state.get("web_context"):
            contexts.append(f"🌐 FROM THE WEB:\n{state['web_context'][:2500]}")
        if state.get("knowledge_context"):
            contexts.append(f"📚 KNOWLEDGE BASE:\n{state['knowledge_context'][:1500]}")
        
        if not contexts:
            contexts.append("No specific context found. Using general knowledge.")
        
        combined = "\n\n---\n\n".join(contexts)
        state["combined_context"] = combined
        
        prompt = f"""You are an AGENTIC AI assistant that synthesizes information from multiple sources.

AVAILABLE CONTEXT:
{combined}

USER QUESTION: {query}

INSTRUCTIONS:
1. Prioritize user's documents (📄) if relevant
2. Add real-time info from web (🌐) when available
3. Use knowledge base (📚) for background
4. Cite sources appropriately
5. Be comprehensive but concise

SYNTHESIZED ANSWER:"""
        
        resp = self.llm.invoke([
            {"role": "system", "content": PPLX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        answer = resp.content
        state["answer"] = answer
        state["followups"] = self.followup.generate(answer, query)
        
        # Combine sources
        all_sources = state.get("file_sources", []) + state.get("web_sources", [])
        state["sources"] = all_sources
        
        print(f"  ✅ AgenticSynthesizerNode: Generated answer with {len(all_sources)} sources")
        return state


# =============================================================================
# ANALYSIS AGENTS
# =============================================================================

class AnalysisSearchNode:
    """Node 1: Search for analysis data."""
    
    def __init__(self):
        self.search_tool = SearchTool()
        self.browse_tool = BrowseTool()
    
    def search(self, state: AnalysisState) -> AnalysisState:
        query = state.get("query", "")
        print(f"  🔍 AnalysisSearchNode: Searching for analysis data")
        
        try:
            results = self.search_tool.search(query, num_results=6)
            state["web_results"] = results
            
            # Fetch content
            web_parts = []
            links = []
            for r in results:
                url = r.get("url")
                title = r.get("title", "")
                if not url:
                    continue
                content = self.browse_tool.fetch_clean(url)
                if content:
                    web_parts.append(f"[{title}]:\n{content[:2000]}")
                    links.append({"title": title, "url": url, "snippet": content[:200]})
            
            state["web_context"] = "\n\n".join(web_parts)
            state["links"] = links
            state["sources"] = [{"title": l["title"], "url": l["url"]} for l in links]
            
        except Exception as e:
            print(f"  ❌ AnalysisSearchNode error: {e}")
            state["web_context"] = ""
            state["links"] = []
            state["sources"] = []
        
        return state


class AnalysisProcessNode:
    """Node 2: Generate structured analysis."""
    
    def __init__(self):
        self.llm = Config.get_llm()
        self.followup = FollowUpGenerator()
    
    def analyze(self, state: AnalysisState) -> AnalysisState:
        query = state.get("query", "")
        context = state.get("web_context", "")
        
        prompt = f"""You are an expert analyst. Provide deep, comprehensive analysis.

RESEARCH DATA:
{context if context else "No external data available."}

ANALYSIS REQUEST: {query}

Provide structured analysis with:

## Executive Summary
(2-3 sentence overview)

## Key Findings
(Bullet points of main discoveries)

## Detailed Analysis
(In-depth examination with evidence)

## Data & Statistics
(Numbers, trends, comparisons if available)

## Conclusions
(Main takeaways)

## Recommendations
(Actionable suggestions)

Use citations [1], [2] when referencing sources.

ANALYSIS:"""
        
        resp = self.llm.invoke([
            {"role": "system", "content": PPLX_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        answer = resp.content
        state["answer"] = answer
        state["followups"] = self.followup.generate(answer, query)
        
        print(f"  ✅ AnalysisProcessNode: Generated analysis")
        return state


# =============================================================================
# SUMMARIZE AGENTS
# =============================================================================

class SummarizeInputNode:
    """Node 1: Determine input type and fetch content."""
    
    def __init__(self):
        self.browse_tool = BrowseTool()
        self.search_tool = SearchTool()
    
    def process_input(self, state: SummarizeState) -> SummarizeState:
        query = state.get("query", "")
        
        # Check if URL
        if query.startswith("http"):
            state["is_url"] = True
            try:
                content = self.browse_tool.fetch_clean(query)
                state["content"] = content or ""
                state["links"] = [{"title": "Source", "url": query, "snippet": content[:200] if content else ""}]
                state["sources"] = [{"title": "Source URL", "url": query}]
                print(f"  🔗 SummarizeInputNode: Fetched URL content")
            except Exception as e:
                print(f"  ❌ Error fetching URL: {e}")
                state["content"] = ""
        else:
            state["is_url"] = False
            # Search and fetch
            try:
                results = self.search_tool.search(query, num_results=3)
                content_parts = []
                links = []
                for r in results:
                    url = r.get("url")
                    title = r.get("title", "")
                    if url:
                        text = self.browse_tool.fetch_clean(url)
                        if text:
                            content_parts.append(text[:1500])
                            links.append({"title": title, "url": url, "snippet": text[:150]})
                
                state["content"] = "\n\n".join(content_parts)
                state["links"] = links
                state["sources"] = [{"title": l["title"], "url": l["url"]} for l in links]
                print(f"  🔍 SummarizeInputNode: Fetched {len(links)} sources")
            except Exception as e:
                print(f"  ❌ Error searching: {e}")
                state["content"] = query  # Use query as content
                state["links"] = []
                state["sources"] = []
        
        return state


class SummarizeProcessNode:
    """Node 2: Generate summary."""
    
    def __init__(self):
        self.summarizer = SummarizerTool()
        self.followup = FollowUpGenerator()
    
    def summarize(self, state: SummarizeState) -> SummarizeState:
        content = state.get("content", "")
        query = state.get("query", "")
        
        if content:
            summary = self.summarizer.summarize(content, max_words=300)
            state["answer"] = summary
        else:
            state["answer"] = "Could not find content to summarize."
        
        state["followups"] = self.followup.generate(state["answer"], query)
        
        print(f"  ✅ SummarizeProcessNode: Generated summary")
        return state

