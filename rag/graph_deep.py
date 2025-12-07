"""
Production-Level LangGraph Pipelines for Perplexity Clone
==========================================================
Each mode has its own graph with proper node structure.
"""

from langgraph.graph import StateGraph, END
from rag.rag_state import (
    RAGState,
    WebSearchState,
    RAGOnlyState,
    AgenticState,
    AnalysisState,
    SummarizeState
)
from rag.agents import (
    # Deep Research agents
    PlannerAgent,
    ResearchAgent,
    AggregatorAgent,
    WriterAgent,
    ValidatorAgent,
    # Web Search agents
    WebSearchNode,
    WebFetchNode,
    WebContextNode,
    WebAnswerNode,
    # RAG agents
    RAGRetrieveNode,
    RAGContextNode,
    RAGAnswerNode,
    # Agentic agents
    AgenticPlannerNode,
    AgenticFileNode,
    AgenticWebNode,
    AgenticKnowledgeNode,
    AgenticImageNode,
    AgenticSynthesizerNode,
    # Analysis agents
    AnalysisSearchNode,
    AnalysisProcessNode,
    # Summarize agents
    SummarizeInputNode,
    SummarizeProcessNode,
)
from vectorstore.store import VectorStore


class DeepResearchGraph:
    """
    Deep Research Mode Graph
    ========================
    Pipeline: Planner → Research → Aggregate → Write → Validate
    
    Used for complex queries requiring multi-step analysis.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent(self.vs)
        self.aggregator = AggregatorAgent()
        self.writer = WriterAgent()
        self.validator = ValidatorAgent()
        self.graph = None

    def build(self):
        g = StateGraph(RAGState)
        g.add_node("plan", self.planner.plan)
        g.add_node("research", self.researcher.research)
        g.add_node("aggregate", self.aggregator.aggregate)
        g.add_node("write", self.writer.write)
        g.add_node("validate", self.validator.validate_and_attach)

        g.set_entry_point("plan")
        g.add_edge("plan", "research")
        g.add_edge("research", "aggregate")
        g.add_edge("aggregate", "write")
        g.add_edge("write", "validate")
        g.add_edge("validate", END)

        self.graph = g.compile()
        return self.graph

    def run(self, question: str) -> RAGState:
        if self.graph is None:
            self.build()
        print(f"\n🧠 DEEP RESEARCH GRAPH: {question[:50]}...")
        return self.graph.invoke({"question": question})


class WebSearchGraph:
    """
    Web Search Mode Graph
    =====================
    Pipeline: Search → Fetch → Context → Answer
    
    Used for real-time web queries with citations.
    """
    
    def __init__(self):
        self.search_node = WebSearchNode()
        self.fetch_node = WebFetchNode()
        self.context_node = WebContextNode()
        self.answer_node = WebAnswerNode()
        self.graph = None
    
    def build(self):
        g = StateGraph(WebSearchState)
        
        g.add_node("search", self.search_node.search)
        g.add_node("fetch", self.fetch_node.fetch)
        g.add_node("build_context", self.context_node.build_context)
        g.add_node("answer", self.answer_node.answer)
        
        g.set_entry_point("search")
        g.add_edge("search", "fetch")
        g.add_edge("fetch", "build_context")
        g.add_edge("build_context", "answer")
        g.add_edge("answer", END)
        
        self.graph = g.compile()
        return self.graph
    
    def run(self, query: str) -> WebSearchState:
        if self.graph is None:
            self.build()
        print(f"\n🌐 WEB SEARCH GRAPH: {query[:50]}...")
        return self.graph.invoke({"query": query})


class RAGOnlyGraph:
    """
    RAG-Only Mode Graph
    ===================
    Pipeline: Retrieve → Context → Answer
    
    Used for searching uploaded documents only.
    """
    
    def __init__(self, file_manager):
        self.retrieve_node = RAGRetrieveNode(file_manager)
        self.context_node = RAGContextNode()
        self.answer_node = RAGAnswerNode()
        self.graph = None
    
    def build(self):
        g = StateGraph(RAGOnlyState)
        
        g.add_node("retrieve", self.retrieve_node.retrieve)
        g.add_node("build_context", self.context_node.build_context)
        g.add_node("answer", self.answer_node.answer)
        
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "build_context")
        g.add_edge("build_context", "answer")
        g.add_edge("answer", END)
        
        self.graph = g.compile()
        return self.graph
    
    def run(self, query: str, workspace_id: str = "default") -> RAGOnlyState:
        if self.graph is None:
            self.build()
        print(f"\n📚 RAG ONLY GRAPH: {query[:50]}...")
        return self.graph.invoke({"query": query, "workspace_id": workspace_id})


class AgenticRAGGraph:
    """
    Agentic RAG Mode Graph
    ======================
    Pipeline: Planner → [File, Web, Knowledge, Image] → Synthesizer
    
    Multi-agent collaboration for comprehensive answers.
    Planner decides which agents to activate.
    """
    
    def __init__(self, file_manager, vector_store: VectorStore, image_search):
        self.planner_node = AgenticPlannerNode()
        self.file_node = AgenticFileNode(file_manager)
        self.web_node = AgenticWebNode()
        self.knowledge_node = AgenticKnowledgeNode(vector_store)
        self.image_node = AgenticImageNode(image_search)
        self.synthesizer_node = AgenticSynthesizerNode()
        self.graph = None
    
    def build(self):
        g = StateGraph(AgenticState)
        
        # Add all nodes
        g.add_node("planner", self.planner_node.plan)
        g.add_node("file_agent", self.file_node.retrieve)
        g.add_node("web_agent", self.web_node.search)
        g.add_node("knowledge_agent", self.knowledge_node.retrieve)
        g.add_node("image_agent", self.image_node.search)
        g.add_node("synthesizer", self.synthesizer_node.synthesize)
        
        # Define flow
        g.set_entry_point("planner")
        
        # After planner, run all agents (they check flags internally)
        g.add_edge("planner", "file_agent")
        g.add_edge("file_agent", "web_agent")
        g.add_edge("web_agent", "knowledge_agent")
        g.add_edge("knowledge_agent", "image_agent")
        g.add_edge("image_agent", "synthesizer")
        g.add_edge("synthesizer", END)
        
        self.graph = g.compile()
        return self.graph
    
    def run(self, query: str, workspace_id: str = "default") -> AgenticState:
        if self.graph is None:
            self.build()
        print(f"\n🤖 AGENTIC RAG GRAPH: {query[:50]}...")
        return self.graph.invoke({"query": query, "workspace_id": workspace_id})


class AnalysisGraph:
    """
    Analysis Mode Graph
    ===================
    Pipeline: Search → Analyze
    
    Deep analysis with structured output format.
    """
    
    def __init__(self):
        self.search_node = AnalysisSearchNode()
        self.process_node = AnalysisProcessNode()
        self.graph = None
    
    def build(self):
        g = StateGraph(AnalysisState)
        
        g.add_node("search", self.search_node.search)
        g.add_node("analyze", self.process_node.analyze)
        
        g.set_entry_point("search")
        g.add_edge("search", "analyze")
        g.add_edge("analyze", END)
        
        self.graph = g.compile()
        return self.graph
    
    def run(self, query: str) -> AnalysisState:
        if self.graph is None:
            self.build()
        print(f"\n📊 ANALYSIS GRAPH: {query[:50]}...")
        return self.graph.invoke({"query": query})


class SummarizeGraph:
    """
    Summarize Mode Graph
    ====================
    Pipeline: Input → Summarize
    
    Handles URL or search-based summarization.
    """
    
    def __init__(self):
        self.input_node = SummarizeInputNode()
        self.process_node = SummarizeProcessNode()
        self.graph = None
    
    def build(self):
        g = StateGraph(SummarizeState)
        
        g.add_node("input", self.input_node.process_input)
        g.add_node("summarize", self.process_node.summarize)
        
        g.set_entry_point("input")
        g.add_edge("input", "summarize")
        g.add_edge("summarize", END)
        
        self.graph = g.compile()
        return self.graph
    
    def run(self, query: str) -> SummarizeState:
        if self.graph is None:
            self.build()
        print(f"\n📝 SUMMARIZE GRAPH: {query[:50]}...")
        return self.graph.invoke({"query": query})
