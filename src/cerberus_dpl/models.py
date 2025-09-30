from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal, List, Dict, Union, Callable, Sequence
from datetime import datetime

# Type alias for adapters
SourceType = Literal["web", "graphql", "database", "cli"]
# Type alias for cluster level statistics
Number = Union[int, float]
# Type alias for token counter utility
TokenCounter = Callable[[Sequence[str]], List[int]]


class NormalizedDoc(BaseModel):
    """Contract for source data."""

    source_type: SourceType
    source_id: str
    timestamp: datetime
    uri: Optional[str] = None
    version_tag: Optional[str] = None
    license: Optional[str] = None

    mime_type: str
    lang: Optional[str] = None
    title: Optional[str] = None
    text: str

    # Structure & links -- mostly for web data, but can be useful in other places
    outlinks: List[str] = []
    backlinks: List[str] = []
    headings: List[str] = []
    code_blocks: int = 0
    tables: int = 0

    last_modified: Optional[datetime] = None
    meta: Dict[str, str] = {}


class DocRecord(BaseModel):
    """Represents a single document's cluster assignment and token length."""

    model_config = ConfigDict(frozen=True)
    doc_index: int
    topic: int
    token_len: int


class TopicStats(BaseModel):
    """Aggregate statistics of token lengths for a given topic."""

    model_config = ConfigDict(frozen=True)
    topic: int
    n_docs: int
    mean_len: Number
    median_len: Number
    min_len: Number
    max_len: Number
    std_len: Number
    iqr: Number
    mad: Number


class OverallSummary(BaseModel):
    """Top-level schema for serializing per-topic stats and metadata."""

    schema_id: str = Field(default="bertopic_cluster_token_stats@v1")
    metadata: dict = Field(default_factory=dict)
    topics: List[TopicStats]
