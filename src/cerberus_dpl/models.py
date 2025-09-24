from pydantic import BaseModel
from typing import Optional, Literal, List, Dict
from datetime import datetime

SourceType = Literal["web", "graphql", "database", "cli"]


class NormalizedDoc(BaseModel):
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
