from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict, total=False):
    """State object for deep research pipeline."""

    question: str
    sub_questions: List[str]
    local_docs: List[Dict[str, Any]]  # Serializable version of docs
    web_pages: List[Dict]              # {"title","url","content"}
    evidence: List[str]                # text snippets
    draft_answers: List[str]           # per sub-question
    final_answer: str
    sources: List[Dict]                # {"title","url"}


class WebSearchState(TypedDict, total=False):
    """State for Web Search mode graph."""
    query: str
    search_results: List[Dict]
    web_pages: List[Dict]
    context: str
    answer: str
    sources: List[Dict]
    links: List[Dict]
    images: List[Dict]
    followups: List[str]


class RAGOnlyState(TypedDict, total=False):
    """State for RAG-only mode graph."""
    query: str
    workspace_id: str
    file_chunks: List[Dict]
    base_chunks: List[Dict]
    context: str
    answer: str
    sources: List[Dict]
    followups: List[str]


class AgenticState(TypedDict, total=False):
    """State for Agentic RAG mode graph."""
    query: str
    workspace_id: str
    
    # Planner outputs
    use_file: bool
    use_web: bool
    use_images: bool
    use_knowledge: bool
    
    # Agent outputs
    file_context: str
    file_sources: List[Dict]
    web_context: str
    web_sources: List[Dict]
    links: List[Dict]
    knowledge_context: str
    images: List[Dict]
    
    # Synthesizer output
    combined_context: str
    answer: str
    sources: List[Dict]
    followups: List[str]


class AnalysisState(TypedDict, total=False):
    """State for Analysis mode graph."""
    query: str
    web_results: List[Dict]
    web_context: str
    analysis: str
    executive_summary: str
    key_findings: List[str]
    answer: str
    sources: List[Dict]
    links: List[Dict]
    images: List[Dict]
    followups: List[str]


class SummarizeState(TypedDict, total=False):
    """State for Summarize mode graph."""
    query: str
    is_url: bool
    content: str
    summary: str
    answer: str
    sources: List[Dict]
    links: List[Dict]
    followups: List[str]

