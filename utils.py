# utils.py

from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class NodeType(str, Enum):
    CONCEPT = "CONCEPT"
    WORKED_EXAMPLE = "WORKED_EXAMPLE"
    PRACTICE_PROBLEM = "PRACTICE_PROBLEM"
    UNKNOWN = "UNKNOWN"

class EdgeType(str, Enum):
    CONTAINS = "CONTAINS"
    EXPLAINS = "EXPLAINS"
    IS_PREREQUISITE_FOR = "IS_PREREQUISITE_FOR"
    # Future extension:
    # IS_GENERALIZED_BY = "IS_GENERALIZED_BY"

class RawNode(BaseModel):
    """
    A temporary data structure for a node before it's added to the graph.
    Represents one semantically distinct chunk from the source document.
    """
    id: str = Field(..., description="A unique identifier for the chunk, e.g., 'chunk_0'.")
    node_type: NodeType = Field(..., description="The classified type of the node.")
    title: str = Field(..., description="A concise, descriptive title for the content chunk.")
    content: str = Field(..., description="The full text content of the chunk.")

class LLMClassificationResponse(BaseModel):
    """Pydantic model for validating the output of Phase 1 LLM call."""
    nodes: List[RawNode]

class GraphEdge(BaseModel):
    """Pydantic model for an edge to be added to the graph."""
    source: str
    target: str
    type: EdgeType

class LLMGraphResponse(BaseModel):
    """Pydantic model for validating the output of Phase 2 LLM call."""
    edges: List[GraphEdge]
