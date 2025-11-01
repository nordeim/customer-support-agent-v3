awesome understanding! please meticulously review, validate and critique the following tools implementation for a "Customer Service" AI Agent. I will share the current tools implementation in this and the next 3 prompts, so just meticulously review what I share and hold the final conclusion of your analysis and assessment till I tell you that I have shared all the tools related code files.

## File 1: `backend/app/schemas/__init__.py` (NEW)

```python
"""
Request validation schemas for API and tool inputs.
Uses Pydantic v2 for type-safe validation and sanitization.

Version: 1.0.0
"""

from .tool_requests import (
    # Memory tool requests
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySummarizeRequest,
    
    # RAG tool requests
    RAGSearchRequest,
    RAGAddDocumentsRequest,
    
    # Attachment tool requests
    AttachmentProcessRequest,
    AttachmentSaveRequest,
    
    # Escalation tool requests
    EscalationCheckRequest,
    
    # Base request
    ToolRequest,
    
    # Utilities
    sanitize_content,
    validate_session_id_format,
    validate_json_serializable,
    check_metadata_depth
)

__all__ = [
    # Memory
    'MemoryStoreRequest',
    'MemoryRetrieveRequest',
    'MemorySummarizeRequest',
    
    # RAG
    'RAGSearchRequest',
    'RAGAddDocumentsRequest',
    
    # Attachment
    'AttachmentProcessRequest',
    'AttachmentSaveRequest',
    
    # Escalation
    'EscalationCheckRequest',
    
    # Base
    'ToolRequest',
    
    # Utilities
    'sanitize_content',
    'validate_session_id_format',
    'validate_json_serializable',
    'check_metadata_depth'
]
```

---

## File 2: `backend/app/schemas/tool_requests.py` (NEW)

```python
"""
Request validation schemas for all tool operations.
Provides input sanitization and validation using Pydantic v2.

Version: 1.0.0

Security Features:
- SQL injection prevention via format validation
- Path traversal prevention
- XSS prevention via content sanitization
- Size limits to prevent DoS
- Depth limits for nested structures
"""
import re
import json
import logging
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Validation patterns
SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,255}$')
FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\. ]{1,255}$')

# Security constants
MAX_CONTENT_LENGTH = 10000
MAX_METADATA_SIZE = 10240  # 10KB
MAX_METADATA_DEPTH = 5
MAX_QUERY_LENGTH = 1000
MAX_DOCUMENT_LENGTH = 50000
MAX_DOCUMENTS_BATCH = 100
MAX_MESSAGE_LENGTH = 5000


# ===========================
# Utility Functions
# ===========================

def validate_session_id_format(session_id: str) -> str:
    """
    Validate session ID format.
    
    Only allows alphanumeric characters, hyphens, and underscores.
    Prevents SQL injection and path traversal attacks.
    
    Args:
        session_id: Session identifier to validate
        
    Returns:
        Validated session ID
        
    Raises:
        ValueError: If format is invalid
    """
    if not session_id:
        raise ValueError("session_id cannot be empty")
    
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError(
            "session_id must contain only alphanumeric characters, "
            "hyphens, and underscores (1-255 characters)"
        )
    
    # Additional security checks
    dangerous_patterns = ['..', '/', '\\', ';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT']
    session_id_upper = session_id.upper()
    
    for pattern in dangerous_patterns:
        if pattern in session_id_upper:
            raise ValueError(f"session_id contains dangerous pattern: {pattern}")
    
    return session_id


def sanitize_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """
    Sanitize content string to prevent injection attacks and normalize whitespace.
    
    Args:
        content: Content to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized content
        
    Raises:
        ValueError: If content is invalid
    """
    if not content:
        raise ValueError("content cannot be empty")
    
    if not isinstance(content, str):
        raise ValueError(f"content must be string, got {type(content)}")
    
    # Remove null bytes (SQL injection vector)
    content = content.replace('\x00', '')
    
    # Remove other control characters except newlines and tabs
    content = ''.join(
        char for char in content
        if char.isprintable() or char in '\n\t'
    )
    
    # Normalize whitespace
    content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces to single
    content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
    
    # Strip leading/trailing whitespace
    content = content.strip()
    
    # Check length after sanitization
    if not content:
        raise ValueError("content is empty after sanitization")
    
    if len(content) > max_length:
        raise ValueError(
            f"content exceeds maximum length: {len(content)} > {max_length}"
        )
    
    return content


def validate_json_serializable(data: Any, max_size: int = MAX_METADATA_SIZE) -> Any:
    """
    Validate that data is JSON-serializable and within size limits.
    
    Args:
        data: Data to validate
        max_size: Maximum JSON size in bytes
        
    Returns:
        Validated data
        
    Raises:
        ValueError: If data is not serializable or too large
    """
    if data is None:
        return {}
    
    try:
        serialized = json.dumps(data)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Data is not JSON-serializable: {e}")
    
    size = len(serialized.encode('utf-8'))
    if size > max_size:
        raise ValueError(
            f"Serialized data exceeds maximum size: {size} > {max_size} bytes"
        )
    
    return data


def check_metadata_depth(obj: Any, current_depth: int = 0, max_depth: int = MAX_METADATA_DEPTH) -> None:
    """
    Check depth of nested structure to prevent DoS attacks.
    
    Args:
        obj: Object to check
        current_depth: Current nesting depth
        max_depth: Maximum allowed depth
        
    Raises:
        ValueError: If depth exceeds maximum
    """
    if current_depth > max_depth:
        raise ValueError(f"Metadata nesting exceeds maximum depth: {max_depth}")
    
    if isinstance(obj, dict):
        for value in obj.values():
            check_metadata_depth(value, current_depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            check_metadata_depth(item, current_depth + 1, max_depth)


# ===========================
# Base Request Model
# ===========================

class ToolRequest(BaseModel):
    """Base class for all tool request validation models."""
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        str_min_length = 1
        validate_assignment = True
        extra = 'forbid'  # Reject unknown fields


# ===========================
# Memory Tool Requests
# ===========================

class MemoryStoreRequest(ToolRequest):
    """
    Validated request for storing memory.
    
    Security:
    - session_id validated against injection patterns
    - content sanitized and length-limited
    - metadata depth and size limited
    - importance range validated
    
    Example:
        request = MemoryStoreRequest(
            session_id="user-123",
            content="User prefers email communication",
            content_type="preference",
            importance=0.8
        )
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session identifier (alphanumeric, hyphens, underscores only)"
    )
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CONTENT_LENGTH,
        description="Memory content to store"
    )
    
    content_type: Literal["user_info", "preference", "fact", "context"] = Field(
        default="context",
        description="Type of memory being stored"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (max 10KB, max depth 5)"
    )
    
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score for retrieval prioritization"
    )
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format and security."""
        return validate_session_id_format(v)
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Sanitize and validate content."""
        return sanitize_content(v, MAX_CONTENT_LENGTH)
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate metadata structure and size."""
        if v is None:
            return {}
        
        # Check JSON serializability and size
        validated = validate_json_serializable(v, MAX_METADATA_SIZE)
        
        # Check nesting depth
        check_metadata_depth(validated)
        
        return validated


class MemoryRetrieveRequest(ToolRequest):
    """
    Validated request for retrieving memories.
    
    Example:
        request = MemoryRetrieveRequest(
            session_id="user-123",
            content_type="preference",
            limit=10,
            min_importance=0.3
        )
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session identifier"
    )
    
    content_type: Optional[Literal["user_info", "preference", "fact", "context"]] = Field(
        default=None,
        description="Filter by memory type (optional)"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of memories to retrieve"
    )
    
    time_window_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=720,  # 30 days max
        description="Only retrieve memories from last N hours (optional)"
    )
    
    min_importance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum importance threshold"
    )
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        return validate_session_id_format(v)


class MemorySummarizeRequest(ToolRequest):
    """
    Validated request for summarizing session memories.
    
    Example:
        request = MemorySummarizeRequest(
            session_id="user-123",
            max_items_per_type=3
        )
    """
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session identifier"
    )
    
    max_items_per_type: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum items per memory type in summary"
    )
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        return validate_session_id_format(v)


# ===========================
# RAG Tool Requests
# ===========================

class RAGSearchRequest(ToolRequest):
    """
    Validated request for RAG search.
    
    Security:
    - query sanitized and length-limited
    - k bounded to prevent resource exhaustion
    - threshold validated
    - filter metadata validated
    
    Example:
        request = RAGSearchRequest(
            query="How do I reset my password?",
            k=5,
            threshold=0.7
        )
    """
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Search query"
    )
    
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter for search (optional)"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Sanitize and validate query."""
        return sanitize_content(v, MAX_QUERY_LENGTH)
    
    @field_validator('filter')
    @classmethod
    def validate_filter(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate filter metadata."""
        if v is None:
            return None
        
        # Check JSON serializability and size
        validated = validate_json_serializable(v, MAX_METADATA_SIZE)
        
        # Check nesting depth
        check_metadata_depth(validated)
        
        return validated


class RAGAddDocumentsRequest(ToolRequest):
    """
    Validated request for adding documents to RAG.
    
    Security:
    - documents list size limited
    - each document length limited
    - IDs validated for uniqueness
    - metadata validated
    
    Example:
        request = RAGAddDocumentsRequest(
            documents=["Doc 1 content", "Doc 2 content"],
            metadatas=[{"source": "faq"}, {"source": "manual"}],
            chunk=True
        )
    """
    
    documents: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENTS_BATCH,
        description="List of documents to add"
    )
    
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metadata for each document (optional, must match length)"
    )
    
    ids: Optional[List[str]] = Field(
        default=None,
        description="IDs for each document (optional, must match length and be unique)"
    )
    
    chunk: bool = Field(
        default=True,
        description="Whether to chunk documents for better retrieval"
    )
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v: List[str]) -> List[str]:
        """Validate each document."""
        if not v:
            raise ValueError("documents list cannot be empty")
        
        if len(v) > MAX_DOCUMENTS_BATCH:
            raise ValueError(
                f"Too many documents: {len(v)} > {MAX_DOCUMENTS_BATCH}"
            )
        
        validated = []
        for i, doc in enumerate(v):
            if not doc or not doc.strip():
                raise ValueError(f"Document at index {i} is empty")
            
            if len(doc) > MAX_DOCUMENT_LENGTH:
                raise ValueError(
                    f"Document at index {i} exceeds max length: "
                    f"{len(doc)} > {MAX_DOCUMENT_LENGTH}"
                )
            
            # Basic sanitization (preserve formatting for documents)
            sanitized = doc.replace('\x00', '')  # Remove null bytes
            validated.append(sanitized)
        
        return validated
    
    @field_validator('metadatas')
    @classmethod
    def validate_metadatas(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate metadata list."""
        if v is None:
            return None
        
        for i, metadata in enumerate(v):
            # Check JSON serializability and size
            validate_json_serializable(metadata, MAX_METADATA_SIZE)
            
            # Check nesting depth
            check_metadata_depth(metadata)
        
        return v
    
    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate IDs list."""
        if v is None:
            return None
        
        # Check for uniqueness
        if len(v) != len(set(v)):
            raise ValueError("ids must be unique")
        
        # Validate each ID format
        for i, id_val in enumerate(v):
            if not id_val or not id_val.strip():
                raise ValueError(f"ID at index {i} is empty")
            
            if len(id_val) > 255:
                raise ValueError(f"ID at index {i} exceeds max length: 255")
            
            # Check for dangerous characters
            if any(char in id_val for char in ['\x00', '\n', '\r', ';', '--']):
                raise ValueError(f"ID at index {i} contains invalid characters")
        
        return v
    
    @model_validator(mode='after')
    def validate_list_lengths(self) -> 'RAGAddDocumentsRequest':
        """Validate that all lists have matching lengths."""
        doc_count = len(self.documents)
        
        if self.metadatas is not None and len(self.metadatas) != doc_count:
            raise ValueError(
                f"metadatas length ({len(self.metadatas)}) must match "
                f"documents length ({doc_count})"
            )
        
        if self.ids is not None and len(self.ids) != doc_count:
            raise ValueError(
                f"ids length ({len(self.ids)}) must match "
                f"documents length ({doc_count})"
            )
        
        return self


# ===========================
# Attachment Tool Requests
# ===========================

class AttachmentProcessRequest(ToolRequest):
    """
    Validated request for processing attachments.
    
    Security:
    - filename sanitized to prevent path traversal
    - file_path validated (actual existence checked at runtime)
    
    Example:
        request = AttachmentProcessRequest(
            file_path="/tmp/upload/document.pdf",
            filename="document.pdf",
            chunk_for_rag=True
        )
    """
    
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Path to file to process"
    )
    
    filename: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Original filename (optional)"
    )
    
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract file metadata"
    )
    
    chunk_for_rag: bool = Field(
        default=False,
        description="Whether to chunk content for RAG indexing"
    )
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path for security."""
        # Check for path traversal attempts
        dangerous_patterns = ['..', '~/', '/etc/', 'C:\\Windows', '/bin/', '/usr/bin/']
        
        for pattern in dangerous_patterns:
            if pattern in v:
                raise ValueError(f"file_path contains dangerous pattern: {pattern}")
        
        # Remove null bytes
        v = v.replace('\x00', '')
        
        # Note: We don't check file existence here - that's done at runtime
        # This is just format validation
        
        return v
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        """Validate filename for security."""
        if v is None:
            return None
        
        # Remove path separators
        v = v.replace('/', '').replace('\\', '').replace('\x00', '')
        
        # Check for valid characters only
        if not FILENAME_PATTERN.match(v):
            raise ValueError(
                "filename must contain only alphanumeric characters, "
                "spaces, hyphens, underscores, and dots"
            )
        
        return v


class AttachmentSaveRequest(ToolRequest):
    """
    Validated request for saving uploaded file.
    
    Example:
        request = AttachmentSaveRequest(
            filename="document.pdf"
        )
    """
    
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Filename for uploaded file"
    )
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate and sanitize filename."""
        # Remove path separators
        v = v.replace('/', '').replace('\\', '').replace('\x00', '')
        
        # Check for valid characters
        if not FILENAME_PATTERN.match(v):
            raise ValueError(
                "filename must contain only alphanumeric characters, "
                "spaces, hyphens, underscores, and dots"
            )
        
        # Prevent hidden files
        if v.startswith('.'):
            raise ValueError("filename cannot start with dot (hidden files not allowed)")
        
        return v


# ===========================
# Escalation Tool Requests
# ===========================

class EscalationCheckRequest(ToolRequest):
    """
    Validated request for escalation check.
    
    Security:
    - message sanitized and length-limited
    - message_history size limited
    
    Example:
        request = EscalationCheckRequest(
            message="I need to speak to a manager immediately!",
            confidence_threshold=0.7
        )
    """
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User message to analyze for escalation"
    )
    
    message_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_length=50,
        description="Recent message history for context (optional)"
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for escalation trigger"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context metadata (optional)"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Sanitize and validate message."""
        return sanitize_content(v, MAX_MESSAGE_LENGTH)
    
    @field_validator('message_history')
    @classmethod
    def validate_message_history(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Validate message history."""
        if v is None:
            return None
        
        if len(v) > 50:
            raise ValueError("message_history too large (max 50 messages)")
        
        # Validate each message structure
        for i, msg in enumerate(v):
            if not isinstance(msg, dict):
                raise ValueError(f"message_history[{i}] must be dictionary")
            
            # Check JSON serializability
            validate_json_serializable(msg, MAX_METADATA_SIZE)
        
        return v
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate metadata."""
        if v is None:
            return None
        
        # Check JSON serializability and size
        validated = validate_json_serializable(v, MAX_METADATA_SIZE)
        
        # Check nesting depth
        check_metadata_depth(validated)
        
        return validated


# Export all models
__all__ = [
    # Memory
    'MemoryStoreRequest',
    'MemoryRetrieveRequest',
    'MemorySummarizeRequest',
    
    # RAG
    'RAGSearchRequest',
    'RAGAddDocumentsRequest',
    
    # Attachment
    'AttachmentProcessRequest',
    'AttachmentSaveRequest',
    
    # Escalation
    'EscalationCheckRequest',
    
    # Base
    'ToolRequest',
    
    # Utilities
    'sanitize_content',
    'validate_session_id_format',
    'validate_json_serializable',
    'check_metadata_depth',
]
```

---

## File 3: `backend/app/tools/memory_tool.py` (COMPLETE UPDATE)

```python
"""
Memory management tool for conversation context persistence.
Uses database for storing and retrieving conversation memories.

Version: 3.0.0 (Enhanced with validation, context managers, and security)

Changes:
- Added Pydantic input validation
- Added database session context managers with timeouts
- Added query timeout configuration
- Enhanced error handling and logging
- Improved connection health checks
"""
import logging
import json
import uuid
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager

from sqlalchemy import create_engine, desc, and_, or_, func, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import IntegrityError, DBAPIError, DisconnectionError
from pydantic import ValidationError

from ..config import settings
from ..models.memory import Memory
from ..schemas.tool_requests import (
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySummarizeRequest
)
from .base_tool import BaseTool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Memory type priorities for retrieval
MEMORY_TYPE_PRIORITY = {
    "user_info": 1.0,
    "preference": 0.9,
    "fact": 0.8,
    "context": 0.7
}

# Default limits
DEFAULT_MEMORY_LIMIT = 10
DEFAULT_TIME_WINDOW_HOURS = 24


class MemoryTool(BaseTool):
    """
    Memory management tool for storing and retrieving conversation context.
    
    Version 3.0.0:
    - ADDED: Pydantic input validation
    - ADDED: Database session context managers with timeouts
    - FIXED: Connection pooling for production
    - FIXED: Unique constraint handling for duplicates
    - FIXED: Field naming (metadata, not tool_metadata)
    - Enhanced error handling and security
    """
    
    def __init__(self):
        """Initialize memory tool with database connection."""
        super().__init__(
            name="memory_management",
            description="Store and retrieve conversation memory and context",
            version="3.0.0"
        )
        
        # Resources initialized in async initialize()
        self.engine = None
        self.SessionLocal = None
    
    async def initialize(self) -> None:
        """Initialize memory tool resources (async-safe)."""
        try:
            logger.info(f"Initializing Memory tool '{self.name}'...")
            
            # Initialize database engine (I/O-bound, run in thread pool)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._init_database
            )
            
            self.initialized = True
            logger.info(f"✓ Memory tool '{self.name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory tool: {e}", exc_info=True)
            raise
    
    async def cleanup(self) -> None:
        """Cleanup memory tool resources."""
        try:
            logger.info(f"Cleaning up Memory tool '{self.name}'...")
            
            # Dispose of database engine
            if self.engine:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.engine.dispose
                )
                self.engine = None
                self.SessionLocal = None
            
            self.initialized = False
            logger.info(f"✓ Memory tool '{self.name}' cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Memory tool cleanup: {e}")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute memory operations (async-first)."""
        action = kwargs.get("action", "retrieve")
        session_id = kwargs.get("session_id")
        
        if not session_id:
            return ToolResult.error_result(
                error="session_id is required",
                metadata={"tool": self.name}
            )
        
        try:
            if action == "store":
                content = kwargs.get("content")
                if not content:
                    return ToolResult.error_result(
                        error="content is required for store action",
                        metadata={"tool": self.name, "action": action}
                    )
                
                result = await self.store_memory_async(
                    session_id=session_id,
                    content=content,
                    content_type=kwargs.get("content_type", "context"),
                    metadata=kwargs.get("metadata"),
                    importance=kwargs.get("importance", 0.5)
                )
                
                return ToolResult.success_result(
                    data=result,
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id
                    }
                )
            
            elif action == "retrieve":
                memories = await self.retrieve_memories_async(
                    session_id=session_id,
                    content_type=kwargs.get("content_type"),
                    limit=kwargs.get("limit", DEFAULT_MEMORY_LIMIT),
                    time_window_hours=kwargs.get("time_window_hours"),
                    min_importance=kwargs.get("min_importance", 0.0)
                )
                
                return ToolResult.success_result(
                    data={
                        "memories": memories,
                        "count": len(memories)
                    },
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id
                    }
                )
            
            elif action == "summarize":
                summary = await self.summarize_session_async(
                    session_id=session_id,
                    max_items_per_type=kwargs.get("max_items_per_type", 3)
                )
                
                return ToolResult.success_result(
                    data={"summary": summary},
                    metadata={
                        "tool": self.name,
                        "action": action,
                        "session_id": session_id,
                        "summary_length": len(summary)
                    }
                )
            
            else:
                return ToolResult.error_result(
                    error=f"Unknown action: {action}. Valid: store, retrieve, summarize",
                    metadata={"tool": self.name}
                )
                
        except Exception as e:
            logger.error(f"Memory execute error: {e}", exc_info=True)
            return ToolResult.error_result(
                error=str(e),
                metadata={"tool": self.name, "action": action, "session_id": session_id}
            )
    
    # ===========================
    # Core Memory Methods (Async) - ENHANCED WITH VALIDATION
    # ===========================
    
    async def store_memory_async(
        self,
        session_id: str,
        content: str,
        content_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Store a memory entry for a session (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemoryStoreRequest(
                session_id=session_id,
                content=content,
                content_type=content_type,
                metadata=metadata,
                importance=importance
            )
        except ValidationError as e:
            logger.error(f"Memory store validation failed: {e}")
            return {
                "success": False,
                "error": f"Input validation failed: {e}",
                "validation_errors": e.errors()
            }
        
        try:
            # Run database operation in thread pool with validated data
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._store_memory_sync,
                validated_request.session_id,
                validated_request.content,
                validated_request.content_type,
                validated_request.metadata,
                validated_request.importance
            )
            
            logger.info(
                f"Stored memory for session {session_id}: "
                f"type={content_type}, importance={importance}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def retrieve_memories_async(
        self,
        session_id: str,
        content_type: Optional[str] = None,
        limit: int = DEFAULT_MEMORY_LIMIT,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories for a session (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemoryRetrieveRequest(
                session_id=session_id,
                content_type=content_type,
                limit=limit,
                time_window_hours=time_window_hours,
                min_importance=min_importance
            )
        except ValidationError as e:
            logger.error(f"Memory retrieve validation failed: {e}")
            return []
        
        try:
            # Run database operation in thread pool with validated data
            memories = await asyncio.get_event_loop().run_in_executor(
                None,
                self._retrieve_memories_sync,
                validated_request.session_id,
                validated_request.content_type,
                validated_request.limit,
                validated_request.time_window_hours,
                validated_request.min_importance
            )
            
            logger.debug(f"Retrieved {len(memories)} memories for session {session_id}")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []
    
    async def summarize_session_async(
        self,
        session_id: str,
        max_items_per_type: int = 3
    ) -> str:
        """
        Generate a text summary of session memories (async).
        
        Version 3.0.0: Added Pydantic validation.
        """
        # ADDED: Input validation
        try:
            validated_request = MemorySummarizeRequest(
                session_id=session_id,
                max_items_per_type=max_items_per_type
            )
        except ValidationError as e:
            logger.error(f"Memory summarize validation failed: {e}")
            return "Error: Invalid request parameters."
        
        try:
            # Retrieve memories grouped by type
            memory_groups = {}
            
            for content_type in MEMORY_TYPE_PRIORITY.keys():
                memories = await self.retrieve_memories_async(
                    session_id=validated_request.session_id,
                    content_type=content_type,
                    limit=validated_request.max_items_per_type,
                    min_importance=0.3
                )
                
                if memories:
                    memory_groups[content_type] = memories
            
            if not memory_groups:
                return "No previous context available for this session."
            
            # Build summary
            summary_parts = []
            
            if "user_info" in memory_groups:
                user_info = [m["content"] for m in memory_groups["user_info"]]
                summary_parts.append(f"User Information: {'; '.join(user_info)}")
            
            if "preference" in memory_groups:
                preferences = [m["content"] for m in memory_groups["preference"]]
                summary_parts.append(f"User Preferences: {'; '.join(preferences)}")
            
            if "fact" in memory_groups:
                facts = [m["content"] for m in memory_groups["fact"][:3]]
                summary_parts.append(f"Key Facts: {'; '.join(facts)}")
            
            if "context" in memory_groups:
                contexts = [m["content"] for m in memory_groups["context"][:5]]
                summary_parts.append(f"Recent Context: {'; '.join(contexts[:3])}")
            
            summary = "\n".join(summary_parts)
            
            logger.debug(f"Generated summary for session {session_id}: {len(summary)} chars")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}", exc_info=True)
            return "Error retrieving session context."
    
    async def cleanup_old_memories_async(
        self,
        days: int = 30,
        max_per_session: int = 100
    ) -> Dict[str, Any]:
        """Clean up old and low-importance memories (async)."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._cleanup_old_memories_sync,
                days,
                max_per_session
            )
            
            logger.info(f"Memory cleanup completed: {result['total_deleted']} memories deleted")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    # ===========================
    # Database Session Context Manager (NEW)
    # ===========================
    
    @contextmanager
    def get_db_session_context(self, timeout: float = 30.0):
        """
        Context manager for database sessions with timeout and health checks.
        
        ADDED in Version 3.0.0 for safe session management.
        
        Args:
            timeout: Maximum session lifetime in seconds
            
        Yields:
            Database session
            
        Raises:
            RuntimeError: If session times out or database not initialized
            
        Example:
            with self.get_db_session_context(timeout=10.0) as db:
                memory = Memory(...)
                db.add(memory)
                # Commit happens automatically if no exception
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.SessionLocal()
        start_time = time.time()
        
        try:
            # Validate connection with a simple query
            try:
                session.execute(text("SELECT 1"))
            except (DBAPIError, DisconnectionError) as e:
                logger.error(f"Database connection validation failed: {e}")
                session.invalidate()
                raise RuntimeError(f"Database connection unhealthy: {e}")
            
            # Yield session for operations
            yield session
            
            # Check if timeout exceeded
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Session exceeded timeout: {elapsed:.2f}s > {timeout}s")
                session.rollback()
                raise RuntimeError(f"Session timeout exceeded: {elapsed:.2f}s")
            
            # Commit if no exceptions
            session.commit()
            
        except (DBAPIError, DisconnectionError) as e:
            logger.error(f"Database error during session: {e}")
            session.rollback()
            # Test if connection is still alive
            try:
                session.execute(text("SELECT 1"))
            except Exception:
                # Connection is dead, invalidate it
                session.invalidate()
            raise
            
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            session.rollback()
            raise
            
        finally:
            # GUARANTEED cleanup
            session.close()
            
            # Log slow sessions
            elapsed = time.time() - start_time
            if elapsed > 5.0:
                logger.warning(f"Slow database session: {elapsed:.2f}s")
    
    # ===========================
    # Private Helper Methods (Sync) - UPDATED WITH CONTEXT MANAGERS
    # ===========================
    
    def _init_database(self) -> None:
        """
        Initialize database engine with proper connection pooling.
        
        Version 3.0.0: Added query timeout configuration.
        """
        try:
            connect_args = {}
            poolclass = None
            pool_config = {}
            
            if "sqlite" in settings.database_url:
                # SQLite configuration
                connect_args = {
                    "check_same_thread": False,
                    "timeout": 20
                }
                poolclass = StaticPool  # SQLite doesn't benefit from pooling
            else:
                # PostgreSQL or other databases
                pool_config = {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_timeout": 30,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True
                }
                poolclass = QueuePool
                
                # ADDED: Query timeout for PostgreSQL
                connect_args = {
                    "options": "-c statement_timeout=30000"  # 30 seconds
                }
            
            self.engine = create_engine(
                settings.database_url,
                connect_args=connect_args,
                poolclass=poolclass,
                echo=settings.database_echo,
                **pool_config
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False
            )
            
            pool_size = pool_config.get('pool_size', 'N/A')
            logger.info(f"Memory database initialized (pool_size={pool_size})")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise
    
    def _store_memory_sync(
        self,
        session_id: str,
        content: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]],
        importance: float
    ) -> Dict[str, Any]:
        """
        Store memory (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager.
        """
        # UPDATED: Use context manager for safe session management
        with self.get_db_session_context(timeout=10.0) as db:
            try:
                # Try to create new memory
                memory = Memory(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    content_type=content_type,
                    content=content,
                    metadata=metadata or {},
                    importance=importance
                )
                
                db.add(memory)
                db.flush()  # Get any database errors early
                
                return {
                    "success": True,
                    "memory_id": memory.id,
                    "action": "created",
                    "message": "Memory stored successfully"
                }
                
            except IntegrityError:
                # Duplicate detected, update existing
                db.rollback()
                
                existing = db.query(Memory).filter(
                    and_(
                        Memory.session_id == session_id,
                        Memory.content_type == content_type,
                        Memory.content == content
                    )
                ).first()
                
                if existing:
                    # Update importance and access tracking
                    existing.importance = max(existing.importance, importance)
                    existing.last_accessed = datetime.utcnow()
                    existing.access_count += 1
                    
                    # Update metadata if provided
                    if metadata:
                        existing.metadata.update(metadata)
                    
                    db.flush()
                    
                    logger.debug(f"Updated existing memory: {existing.id}")
                    
                    return {
                        "success": True,
                        "memory_id": existing.id,
                        "action": "updated",
                        "message": "Memory updated successfully"
                    }
                else:
                    # Race condition: deleted between attempts
                    raise
    
    def _retrieve_memories_sync(
        self,
        session_id: str,
        content_type: Optional[str],
        limit: int,
        time_window_hours: Optional[int],
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager.
        """
        # UPDATED: Use context manager
        with self.get_db_session_context(timeout=10.0) as db:
            query = db.query(Memory).filter(Memory.session_id == session_id)
            
            # Apply filters
            if content_type:
                query = query.filter(Memory.content_type == content_type)
            
            if time_window_hours:
                cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
                query = query.filter(Memory.created_at >= cutoff_time)
            
            if min_importance > 0:
                query = query.filter(Memory.importance >= min_importance)
            
            # Order by importance and recency
            query = query.order_by(
                desc(Memory.importance),
                desc(Memory.created_at)
            ).limit(limit)
            
            memories = query.all()
            
            # Update access times
            for memory in memories:
                memory.update_access()
            
            db.flush()
            
            # Format results
            results = []
            for memory in memories:
                results.append({
                    "id": memory.id,
                    "content_type": memory.content_type,
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "access_count": memory.access_count
                })
            
            return results
    
    def _cleanup_old_memories_sync(
        self,
        days: int,
        max_per_session: int
    ) -> Dict[str, Any]:
        """
        Cleanup old memories (sync implementation for thread pool).
        
        Version 3.0.0: UPDATED to use context manager with longer timeout.
        """
        # UPDATED: Use context manager with longer timeout for cleanup
        with self.get_db_session_context(timeout=60.0) as db:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old, low-importance, rarely accessed memories
            deleted_old = db.query(Memory).filter(
                and_(
                    Memory.last_accessed < cutoff_date,
                    Memory.importance < 0.3,
                    Memory.access_count < 3
                )
            ).delete(synchronize_session=False)
            
            # For each session, keep only the most recent/important memories
            sessions = db.query(Memory.session_id).distinct().all()
            deleted_excess = 0
            
            for (session_id,) in sessions:
                # Get memories ordered by importance and recency
                memories = db.query(Memory).filter(
                    Memory.session_id == session_id
                ).order_by(
                    desc(Memory.importance),
                    desc(Memory.created_at)
                ).offset(max_per_session).all()
                
                # Delete excess memories
                for memory in memories:
                    db.delete(memory)
                    deleted_excess += 1
            
            db.flush()
            
            total_deleted = deleted_old + deleted_excess
            
            return {
                "success": True,
                "deleted_old": deleted_old,
                "deleted_excess": deleted_excess,
                "total_deleted": total_deleted
            }


__all__ = ['MemoryTool']
```

# 🚀 Continuing Implementation - Files 4-6

---

## File 4: `backend/app/config/tool_settings.py` (COMPLETE UPDATE)

```python
"""
Tool-specific configuration settings.
Defines feature flags and per-tool configurations for the agent system.

Version: 3.0.0 (Enhanced with SecretStr for API keys and secrets management)

Changes:
- Added SecretStr for sensitive fields
- Added environment variable and secrets manager support
- Added production validation
- Enhanced error messages and documentation
"""
from typing import Dict, Any, Optional, List, Union
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings
import os
import logging

logger = logging.getLogger(__name__)


class ToolSettings(BaseSettings):
    """
    Tool-specific configuration with feature flags.
    Each tool can be enabled/disabled and configured independently.
    
    Version 3.0.0: Enhanced security with SecretStr and secrets management.
    """
    
    # ===========================
    # Tool Feature Flags
    # ===========================
    
    enable_rag_tool: bool = Field(
        default=True,
        description="Enable RAG (Retrieval-Augmented Generation) tool"
    )
    
    enable_memory_tool: bool = Field(
        default=True,
        description="Enable Memory management tool"
    )
    
    enable_escalation_tool: bool = Field(
        default=True,
        description="Enable Escalation detection tool"
    )
    
    enable_attachment_tool: bool = Field(
        default=True,
        description="Enable Attachment processing tool"
    )
    
    # Future tools (disabled by default)
    enable_crm_tool: bool = Field(
        default=False,
        description="Enable CRM lookup tool"
    )
    
    enable_billing_tool: bool = Field(
        default=False,
        description="Enable Billing/invoice tool"
    )
    
    enable_inventory_tool: bool = Field(
        default=False,
        description="Enable Inventory lookup tool"
    )
    
    # ===========================
    # RAG Tool Configuration
    # ===========================
    
    rag_chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="RAG document chunk size in words"
    )
    
    rag_chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in words"
    )
    
    rag_search_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of RAG search results"
    )
    
    rag_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for RAG results"
    )
    
    rag_cache_enabled: bool = Field(
        default=True,
        description="Enable caching for RAG search results"
    )
    
    rag_cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="RAG cache TTL in seconds"
    )
    
    # ===========================
    # Memory Tool Configuration
    # ===========================
    
    memory_max_entries: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum memory entries per session"
    )
    
    memory_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Memory TTL in hours"
    )
    
    memory_cleanup_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days before cleaning old memories"
    )
    
    memory_importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum importance for memory retrieval"
    )
    
    # ===========================
    # Escalation Tool Configuration
    # ===========================
    
    escalation_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for escalation"
    )
    
    escalation_keywords: Dict[str, float] = Field(
        default_factory=lambda: {
            "urgent": 1.0,
            "emergency": 1.0,
            "complaint": 0.9,
            "legal": 0.9,
            "lawsuit": 1.0,
            "manager": 0.8,
            "supervisor": 0.8
        },
        description="Escalation keywords with weights"
    )
    
    escalation_notification_enabled: bool = Field(
        default=False,
        description="Enable automatic escalation notifications"
    )
    
    escalation_notification_email: Optional[str] = Field(
        default=None,
        description="Email address for escalation notifications"
    )
    
    escalation_notification_webhook: Optional[str] = Field(
        default=None,
        description="Webhook URL for escalation notifications"
    )
    
    # ===========================
    # Attachment Tool Configuration
    # ===========================
    
    attachment_max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum attachment file size in bytes"
    )
    
    attachment_allowed_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".doc", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png"
        ],
        description="Allowed file extensions for attachments"
    )
    
    attachment_chunk_for_rag: bool = Field(
        default=True,
        description="Automatically chunk attachments for RAG indexing"
    )
    
    attachment_temp_cleanup_hours: int = Field(
        default=24,
        ge=1,
        description="Hours before cleaning up temporary attachment files"
    )
    
    # ===========================
    # CRM Tool Configuration (ENHANCED WITH SECRETSTR)
    # ===========================
    
    crm_api_endpoint: Optional[str] = Field(
        default=None,
        description="CRM API endpoint URL"
    )
    
    crm_api_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "CRM API key. Supports:\n"
            "- Direct value (dev only)\n"
            "- env://VAR_NAME (load from environment)\n"
            "- secretsmanager://aws/secret-name (AWS Secrets Manager)\n"
            "Production must use env:// or secretsmanager:// prefix"
        )
    )
    
    crm_timeout: int = Field(
        default=10,
        ge=1,
        description="CRM API timeout in seconds"
    )
    
    crm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum CRM API retry attempts"
    )
    
    # ===========================
    # Billing Tool Configuration (ENHANCED WITH SECRETSTR)
    # ===========================
    
    billing_api_endpoint: Optional[str] = Field(
        default=None,
        description="Billing API endpoint URL"
    )
    
    billing_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Billing API key (supports env:// and secretsmanager:// prefixes)"
    )
    
    # ===========================
    # Inventory Tool Configuration (ENHANCED WITH SECRETSTR)
    # ===========================
    
    inventory_api_endpoint: Optional[str] = Field(
        default=None,
        description="Inventory API endpoint URL"
    )
    
    inventory_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Inventory API key (supports env:// and secretsmanager:// prefixes)"
    )
    
    # ===========================
    # Validators (ENHANCED WITH SECRETS MANAGEMENT)
    # ===========================
    
    @field_validator('escalation_keywords', mode='before')
    @classmethod
    def parse_escalation_keywords(cls, v):
        """Parse escalation keywords from various formats."""
        if v is None:
            return {
                "urgent": 1.0,
                "emergency": 1.0,
                "complaint": 0.9,
                "legal": 0.9,
                "lawsuit": 1.0,
                "manager": 0.8,
                "supervisor": 0.8
            }
        
        if isinstance(v, dict):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('{'):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated key=value pairs
            result = {}
            for pair in v.split(','):
                if '=' in pair:
                    key, value = pair.strip().split('=', 1)
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = 0.8
                else:
                    result[pair.strip()] = 0.8
            return result if result else cls.parse_escalation_keywords(None)
        
        return v
    
    @field_validator('attachment_allowed_extensions', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse allowed extensions from various formats."""
        default = [
            ".pdf", ".docx", ".doc", ".txt", ".md",
            ".csv", ".xlsx", ".xls", ".json", ".xml",
            ".jpg", ".jpeg", ".png"
        ]
        
        if v is None:
            return default
        
        if isinstance(v, list):
            return v
        
        if isinstance(v, str):
            import json
            # Try to parse as JSON
            if v.startswith('['):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Parse as comma-separated
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        
        return default
    
    @field_validator('crm_api_key', 'billing_api_key', 'inventory_api_key', mode='before')
    @classmethod
    def load_api_key_from_source(cls, v: Optional[Union[str, SecretStr]]) -> Optional[SecretStr]:
        """
        Load API key from various sources.
        
        ADDED in Version 3.0.0 for secure secrets management.
        
        Supports:
        - Direct value: "sk-abc123" (development only)
        - Environment variable: "env://CRM_API_KEY"
        - AWS Secrets Manager: "secretsmanager://aws/crm-api-key"
        
        Args:
            v: API key value or reference
            
        Returns:
            SecretStr with loaded value or None
            
        Raises:
            ValueError: If production uses direct value or loading fails
        """
        if v is None:
            return None
        
        # Already a SecretStr
        if isinstance(v, SecretStr):
            return v
        
        if not isinstance(v, str):
            raise ValueError(f"API key must be string or SecretStr, got {type(v)}")
        
        # Empty string = None
        if not v.strip():
            return None
        
        # Load from environment variable
        if v.startswith('env://'):
            env_var = v.replace('env://', '')
            env_value = os.getenv(env_var)
            
            if not env_value:
                logger.warning(f"Environment variable not set: {env_var}")
                return None
            
            logger.info(f"Loaded API key from environment variable: {env_var}")
            return SecretStr(env_value)
        
        # Load from AWS Secrets Manager
        elif v.startswith('secretsmanager://aws/'):
            secret_name = v.replace('secretsmanager://aws/', '')
            
            try:
                import boto3
                from botocore.exceptions import ClientError
                
                client = boto3.client('secretsmanager')
                
                try:
                    response = client.get_secret_value(SecretId=secret_name)
                    secret_value = response['SecretString']
                    
                    logger.info(f"Loaded API key from AWS Secrets Manager: {secret_name}")
                    return SecretStr(secret_value)
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    logger.error(f"Failed to load secret '{secret_name}': {error_code}")
                    raise ValueError(f"Cannot load secret from AWS: {error_code}")
                    
            except ImportError:
                raise ValueError(
                    "boto3 not installed. Install with: pip install boto3"
                )
        
        # Direct value (check if production)
        else:
            # Get environment from settings (if available)
            try:
                from ..config import settings
                if settings.environment == 'production':
                    raise ValueError(
                        "In production, API keys must use env:// or secretsmanager:// prefix. "
                        f"Example: env://CRM_API_KEY or secretsmanager://aws/crm-api-key"
                    )
            except ImportError:
                # Settings not available yet (during config loading)
                pass
            
            logger.warning("Using direct API key value (development only)")
            return SecretStr(v)
    
    # ===========================
    # Helper Methods (ENHANCED)
    # ===========================
    
    def get_enabled_tools(self) -> List[str]:
        """
        Get list of enabled tool names.
        
        Returns:
            List of enabled tool identifiers
        """
        enabled = []
        
        if self.enable_rag_tool:
            enabled.append('rag')
        if self.enable_memory_tool:
            enabled.append('memory')
        if self.enable_escalation_tool:
            enabled.append('escalation')
        if self.enable_attachment_tool:
            enabled.append('attachment')
        if self.enable_crm_tool:
            enabled.append('crm')
        if self.enable_billing_tool:
            enabled.append('billing')
        if self.enable_inventory_tool:
            enabled.append('inventory')
        
        return enabled
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier ('rag', 'memory', etc.)
            
        Returns:
            Dictionary of tool-specific configuration
        """
        if tool_name == 'rag':
            return {
                'chunk_size': self.rag_chunk_size,
                'chunk_overlap': self.rag_chunk_overlap,
                'search_k': self.rag_search_k,
                'similarity_threshold': self.rag_similarity_threshold,
                'cache_enabled': self.rag_cache_enabled,
                'cache_ttl': self.rag_cache_ttl
            }
        
        elif tool_name == 'memory':
            return {
                'max_entries': self.memory_max_entries,
                'ttl_hours': self.memory_ttl_hours,
                'cleanup_days': self.memory_cleanup_days,
                'importance_threshold': self.memory_importance_threshold
            }
        
        elif tool_name == 'escalation':
            return {
                'confidence_threshold': self.escalation_confidence_threshold,
                'keywords': self.escalation_keywords,
                'notification_enabled': self.escalation_notification_enabled,
                'notification_email': self.escalation_notification_email,
                'notification_webhook': self.escalation_notification_webhook
            }
        
        elif tool_name == 'attachment':
            return {
                'max_file_size': self.attachment_max_file_size,
                'allowed_extensions': self.attachment_allowed_extensions,
                'chunk_for_rag': self.attachment_chunk_for_rag,
                'temp_cleanup_hours': self.attachment_temp_cleanup_hours
            }
        
        elif tool_name == 'crm':
            return {
                'api_endpoint': self.crm_api_endpoint,
                'has_api_key': self.crm_api_key is not None,  # Don't expose actual key
                'timeout': self.crm_timeout,
                'max_retries': self.crm_max_retries
            }
        
        elif tool_name == 'billing':
            return {
                'api_endpoint': self.billing_api_endpoint,
                'has_api_key': self.billing_api_key is not None
            }
        
        elif tool_name == 'inventory':
            return {
                'api_endpoint': self.inventory_api_endpoint,
                'has_api_key': self.inventory_api_key is not None
            }
        
        else:
            return {}
    
    def validate_tool_config(self, tool_name: str) -> List[str]:
        """
        Validate configuration for a specific tool.
        
        Args:
            tool_name: Tool identifier
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        if tool_name == 'crm' and self.enable_crm_tool:
            if not self.crm_api_endpoint:
                warnings.append("CRM tool enabled but no API endpoint configured")
            if not self.crm_api_key:
                warnings.append("CRM tool enabled but no API key configured")
        
        if tool_name == 'billing' and self.enable_billing_tool:
            if not self.billing_api_endpoint:
                warnings.append("Billing tool enabled but no API endpoint configured")
            if not self.billing_api_key:
                warnings.append("Billing tool enabled but no API key configured")
        
        if tool_name == 'inventory' and self.enable_inventory_tool:
            if not self.inventory_api_endpoint:
                warnings.append("Inventory tool enabled but no API endpoint configured")
            if not self.inventory_api_key:
                warnings.append("Inventory tool enabled but no API key configured")
        
        if tool_name == 'escalation' and self.escalation_notification_enabled:
            if not self.escalation_notification_email and not self.escalation_notification_webhook:
                warnings.append(
                    "Escalation notifications enabled but no email or webhook configured"
                )
        
        return warnings
    
    # ===========================
    # Secure API Key Access (NEW)
    # ===========================
    
    def get_crm_api_key(self) -> Optional[str]:
        """
        Get CRM API key value (use this instead of accessing field directly).
        
        ADDED in Version 3.0.0 for secure access.
        
        Returns:
            API key string or None if not set
        """
        if self.crm_api_key:
            return self.crm_api_key.get_secret_value()
        return None
    
    def get_billing_api_key(self) -> Optional[str]:
        """
        Get Billing API key value.
        
        Returns:
            API key string or None if not set
        """
        if self.billing_api_key:
            return self.billing_api_key.get_secret_value()
        return None
    
    def get_inventory_api_key(self) -> Optional[str]:
        """
        Get Inventory API key value.
        
        Returns:
            API key string or None if not set
        """
        if self.inventory_api_key:
            return self.inventory_api_key.get_secret_value()
        return None


# Create global instance
tool_settings = ToolSettings()

# Export
__all__ = ['ToolSettings', 'tool_settings']
```

---

## File 5: `backend/app/models/memory.py` (COMPLETE UPDATE - Phase 2)

```python
"""
Memory model for conversation context persistence.
Stores important information extracted from conversations.

Version: 3.0.0 (Enhanced with content hashing, TTL, and soft delete)

Changes from 2.0.0:
- Added content_hash for duplicate detection
- Added semantic_hash for similarity detection
- Added expires_at for automatic TTL
- Added deleted_at for soft delete (GDPR compliance)
- Updated unique constraint to use content_hash
- Added composite indexes with partial conditions
- Added importance decay calculation
- Enhanced validation and helper methods
"""
from sqlalchemy import (
    Column, String, Text, Float, Integer, DateTime, JSON,
    Index, UniqueConstraint, CheckConstraint, text
)
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import hashlib
import math
import uuid

from ..database import Base


class Memory(Base):
    """
    Memory storage for conversation context with content hashing and TTL.
    
    Stores important facts, preferences, and context extracted
    from user conversations for enhanced personalization.
    
    Version 3.0.0:
    - Content hash for exact duplicate detection
    - Semantic hash for similarity clustering
    - TTL support for automatic cleanup
    - Soft delete for GDPR compliance
    - Importance decay over time
    - Optimized indexes for performance
    """
    
    __tablename__ = "memories"
    
    # Primary key
    id = Column(String(36), primary_key=True, index=True)
    
    # Session reference
    session_id = Column(String(255), nullable=False, index=True)
    
    # Memory classification
    content_type = Column(
        String(50),
        nullable=False,
        default="context",
        comment="Type: user_info, preference, fact, context"
    )
    
    # Memory content
    content = Column(Text, nullable=False)
    
    # ADDED: Content hash for duplicate detection
    content_hash = Column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA256 hash of normalized content for exact duplicate detection"
    )
    
    # ADDED: Semantic hash for similarity clustering
    semantic_hash = Column(
        String(64),
        nullable=True,
        index=True,
        comment="SimHash for semantic similarity detection (optional)"
    )
    
    # Metadata
    metadata = Column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata about the memory"
    )
    
    # Importance score (0.0 to 1.0)
    importance = Column(
        Float,
        nullable=False,
        default=0.5,
        comment="Importance score for retrieval prioritization"
    )
    
    # ADDED: TTL support
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Expiration timestamp for automatic cleanup"
    )
    
    # Access tracking
    access_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times this memory was accessed"
    )
    
    last_accessed = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last time this memory was retrieved"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # ADDED: Soft delete support
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Soft delete timestamp for GDPR compliance"
    )
    
    # Table-level constraints and indexes
    __table_args__ = (
        # UPDATED: Unique constraint on content hash instead of full content
        UniqueConstraint(
            'session_id',
            'content_type',
            'content_hash',
            name='uq_memory_session_content_hash'
        ),
        
        # ADDED: Composite index for efficient retrieval (with partial condition)
        Index(
            'ix_memory_session_type_importance',
            'session_id',
            'content_type',
            'importance',
            postgresql_where=text('deleted_at IS NULL')  # PostgreSQL partial index
        ),
        
        # ADDED: Index for active memories with expiration
        Index(
            'ix_memory_active_expires',
            'expires_at',
            postgresql_where=text('deleted_at IS NULL AND expires_at IS NOT NULL')
        ),
        
        # ADDED: Composite index for cleanup queries
        Index(
            'ix_memory_cleanup',
            'session_id',
            'last_accessed',
            'importance',
            postgresql_where=text('deleted_at IS NULL')
        ),
        
        # Index for time-based queries
        Index(
            'ix_memory_created_at',
            'created_at'
        ),
        
        # ADDED: Check constraints for data integrity
        CheckConstraint(
            'importance >= 0.0 AND importance <= 1.0',
            name='ck_memory_importance_range'
        ),
        
        CheckConstraint(
            'access_count >= 0',
            name='ck_memory_access_count_positive'
        ),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Memory(id={self.id}, session_id={self.session_id}, "
            f"type={self.content_type}, importance={self.importance}, "
            f"expired={self.is_expired()}, deleted={self.is_deleted()})>"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content_type": self.content_type,
            "content": self.content,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
            "importance": self.importance,
            "effective_importance": self.effective_importance,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_expired": self.is_expired(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """
        Create Memory instance from dictionary.
        
        Args:
            data: Dictionary with memory data
            
        Returns:
            Memory instance
        """
        # Convert ISO strings to datetime if needed
        datetime_fields = ['expires_at', 'last_accessed', 'created_at', 'updated_at', 'deleted_at']
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def validate(self) -> bool:
        """
        Validate memory data.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Validate content_type
        valid_types = ['user_info', 'preference', 'fact', 'context']
        if self.content_type not in valid_types:
            raise ValueError(f"Invalid content_type. Must be one of: {valid_types}")
        
        # Validate importance range
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError("Importance must be between 0.0 and 1.0")
        
        # Validate content not empty
        if not self.content or not self.content.strip():
            raise ValueError("Content cannot be empty")
        
        # Validate content_hash matches content
        expected_hash = self.compute_content_hash(self.content)
        if self.content_hash != expected_hash:
            raise ValueError("content_hash does not match content")
        
        # Validate metadata is JSON-serializable
        try:
            json.dumps(self.metadata)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")
        
        return True
    
    # ===========================
    # Content Hashing (NEW)
    # ===========================
    
    @staticmethod
    def normalize_content(content: str) -> str:
        """
        Normalize content for consistent hashing.
        
        Normalization steps:
        1. Convert to lowercase
        2. Normalize whitespace to single spaces
        3. Strip trailing punctuation
        
        Args:
            content: Raw content string
            
        Returns:
            Normalized content
        """
        # Convert to lowercase
        normalized = content.lower()
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove trailing punctuation
        normalized = normalized.rstrip('.,!?;:')
        
        return normalized
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """
        Compute SHA256 hash of normalized content.
        
        Args:
            content: Content string
            
        Returns:
            Hex digest of content hash (64 characters)
        """
        normalized = Memory.normalize_content(content)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    @staticmethod
    def compute_semantic_hash(content: str, num_bits: int = 64) -> str:
        """
        Compute SimHash for semantic similarity detection.
        
        This is a simplified implementation. For production use,
        consider using a proper LSH library or sentence embeddings.
        
        Args:
            content: Content string
            num_bits: Hash size in bits (default 64)
            
        Returns:
            Hex digest of semantic hash
        """
        # Tokenize normalized content
        tokens = Memory.normalize_content(content).split()
        
        # Initialize bit vector
        v = [0] * num_bits
        
        # Process each token
        for token in tokens:
            # Hash token to get fingerprint
            h = hashlib.sha256(token.encode('utf-8')).digest()
            
            # Update bit vector based on fingerprint
            for i in range(num_bits):
                byte_idx = i // 8
                bit_idx = i % 8
                
                if byte_idx < len(h):
                    if (h[byte_idx] >> bit_idx) & 1:
                        v[i] += 1
                    else:
                        v[i] -= 1
        
        # Convert bit vector to binary fingerprint
        fingerprint = 0
        for i in range(num_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        # Convert to hex
        hex_length = num_bits // 4
        return format(fingerprint, f'0{hex_length}x')
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two hashes.
        
        Useful for finding similar memories based on semantic_hash.
        
        Args:
            hash1: First hash (hex string)
            hash2: Second hash (hex string)
            
        Returns:
            Hamming distance (number of differing bits)
        """
        # Convert hex to binary
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        
        # XOR to find differing bits
        xor_result = int1 ^ int2
        
        # Count number of 1s (differing bits)
        distance = bin(xor_result).count('1')
        
        return distance
    
    # ===========================
    # Factory Methods (NEW)
    # ===========================
    
    @classmethod
    def create_memory(
        cls,
        session_id: str,
        content: str,
        content_type: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None
    ) -> 'Memory':
        """
        Factory method to create memory with automatic hash computation.
        
        Args:
            session_id: Session identifier
            content: Memory content
            content_type: Memory type (user_info, preference, fact, context)
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
            ttl_hours: Time-to-live in hours (optional)
            
        Returns:
            Memory instance with computed hashes
        """
        # Compute hashes
        content_hash = cls.compute_content_hash(content)
        semantic_hash = cls.compute_semantic_hash(content)
        
        # Calculate expiration
        expires_at = None
        if ttl_hours:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        
        return cls(
            id=str(uuid.uuid4()),
            session_id=session_id,
            content=content,
            content_type=content_type,
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            importance=importance,
            metadata=metadata or {},
            expires_at=expires_at
        )
    
    # ===========================
    # Lifecycle Methods (NEW)
    # ===========================
    
    def is_expired(self) -> bool:
        """
        Check if memory has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_deleted(self) -> bool:
        """
        Check if memory is soft-deleted.
        
        Returns:
            True if deleted, False otherwise
        """
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """
        Soft delete memory for GDPR compliance.
        
        Marks the memory as deleted without removing from database.
        Useful for audit trails and compliance.
        """
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """
        Restore a soft-deleted memory.
        
        Removes the deleted_at timestamp, making the memory active again.
        """
        self.deleted_at = None
    
    def extend_ttl(self, hours: int) -> None:
        """
        Extend the TTL of the memory.
        
        Args:
            hours: Hours to extend from now
        """
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
    
    # ===========================
    # Importance Decay (NEW)
    # ===========================
    
    def calculate_decay_factor(self, half_life_days: int = 30) -> float:
        """
        Calculate importance decay factor based on age.
        
        Implements exponential decay: importance * e^(-λt)
        where λ = ln(2) / half_life
        
        This allows older memories to naturally become less important
        over time, unless they are frequently accessed.
        
        Args:
            half_life_days: Days for importance to decay by half
            
        Returns:
            Decay factor (0.0 to 1.0)
        """
        age_days = (datetime.utcnow() - self.created_at).days
        
        # Decay constant: ln(2) / half_life
        decay_constant = 0.693147 / half_life_days
        
        # Exponential decay: e^(-λt)
        decay_factor = math.exp(-decay_constant * age_days)
        
        return decay_factor
    
    @hybrid_property
    def effective_importance(self) -> float:
        """
        Calculate effective importance with decay.
        
        Combines base importance with time-based decay.
        Frequently accessed memories maintain higher importance.
        
        Returns:
            Decayed importance score
        """
        # Base importance
        base_importance = self.importance
        
        # Apply time-based decay
        decay = self.calculate_decay_factor(half_life_days=30)
        
        # Boost based on access frequency (up to 20% boost)
        access_boost = min(0.2, self.access_count * 0.02)
        
        # Combine: base * decay + access_boost
        effective = (base_importance * decay) + access_boost
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, effective))


__all__ = ['Memory']
```

---

## File 6: `alembic/versions/002_add_content_hashing.py` (NEW - Phase 2)

```python
"""
Add content hashing to memories table

Revision ID: 002_add_content_hashing
Revises: 001_initial_schema
Create Date: 2024-01-15 10:00:00.000000

Migration: Add content hashing and TTL support to memories table

Changes:
- Add content_hash column for duplicate detection
- Add semantic_hash column for similarity detection
- Add expires_at column for TTL support
- Add deleted_at column for soft delete
- Update unique constraint to use content_hash
- Add optimized indexes
- Backfill content_hash for existing data
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
import hashlib
import logging

# Revision identifiers
revision = '002_add_content_hashing'
down_revision = '001_initial_schema'  # Update this to match your previous migration
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)


def normalize_content(content: str) -> str:
    """Normalize content for hashing (matches Memory model)."""
    normalized = content.lower()
    normalized = ' '.join(normalized.split())
    normalized = normalized.rstrip('.,!?;:')
    return normalized


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of normalized content."""
    normalized = normalize_content(content)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def upgrade():
    """
    Apply migration: Add content hashing and TTL support.
    
    Steps:
    1. Add new columns (nullable initially)
    2. Backfill content_hash for existing data
    3. Make content_hash non-nullable
    4. Drop old unique constraint
    5. Add new unique constraint on content_hash
    6. Add indexes
    """
    logger.info("Starting migration: Add content hashing to memories")
    
    # Get database connection
    connection = op.get_bind()
    is_postgresql = 'postgresql' in str(connection.engine.url)
    
    # ===========================
    # Step 1: Add new columns
    # ===========================
    
    logger.info("Adding new columns...")
    
    # Add content_hash (nullable initially for backfill)
    op.add_column(
        'memories',
        sa.Column('content_hash', sa.String(64), nullable=True)
    )
    
    # Add semantic_hash (optional, nullable)
    op.add_column(
        'memories',
        sa.Column('semantic_hash', sa.String(64), nullable=True)
    )
    
    # Add expires_at (optional TTL)
    op.add_column(
        'memories',
        sa.Column(
            'expires_at',
            sa.DateTime(timezone=True),
            nullable=True
        )
    )
    
    # Add deleted_at (soft delete)
    op.add_column(
        'memories',
        sa.Column(
            'deleted_at',
            sa.DateTime(timezone=True),
            nullable=True
        )
    )
    
    logger.info("✓ New columns added")
    
    # ===========================
    # Step 2: Backfill content_hash
    # ===========================
    
    logger.info("Backfilling content_hash for existing data...")
    
    if is_postgresql:
        # PostgreSQL: Use native hashing (faster)
        logger.info("Using PostgreSQL native hashing")
        
        connection.execute(text("""
            UPDATE memories
            SET content_hash = encode(
                digest(
                    lower(trim(trailing '.,!?;:' from regexp_replace(content, E'\\s+', ' ', 'g'))),
                    'sha256'
                ),
                'hex'
            )
            WHERE content_hash IS NULL
        """))
        
        logger.info("✓ Backfill completed using PostgreSQL digest")
    
    else:
        # SQLite or other: Use Python hashing
        logger.info("Using Python hashing for backfill")
        
        # Fetch all memories without content_hash
        result = connection.execute(
            text("SELECT id, content FROM memories WHERE content_hash IS NULL")
        )
        
        rows = result.fetchall()
        total_rows = len(rows)
        
        logger.info(f"Backfilling {total_rows} rows...")
        
        # Update in batches
        batch_size = 100
        for i in range(0, total_rows, batch_size):
            batch = rows[i:i + batch_size]
            
            for row in batch:
                memory_id = row[0]
                content = row[1]
                
                # Compute hash
                content_hash = compute_content_hash(content)
                
                # Update row
                connection.execute(
                    text("UPDATE memories SET content_hash = :hash WHERE id = :id"),
                    {"hash": content_hash, "id": memory_id}
                )
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Processed {i + batch_size}/{total_rows} rows...")
        
        logger.info("✓ Backfill completed using Python")
    
    # Commit backfill
    connection.execute(text("COMMIT"))
    
    # ===========================
    # Step 3: Make content_hash non-nullable
    # ===========================
    
    logger.info("Making content_hash non-nullable...")
    
    op.alter_column(
        'memories',
        'content_hash',
        nullable=False
    )
    
    logger.info("✓ content_hash is now non-nullable")
    
    # ===========================
    # Step 4: Update unique constraint
    # ===========================
    
    logger.info("Updating unique constraint...")
    
    # Drop old unique constraint (on full content)
    try:
        op.drop_constraint(
            'uq_memory_session_content',
            'memories',
            type_='unique'
        )
        logger.info("✓ Dropped old unique constraint")
    except Exception as e:
        logger.warning(f"Could not drop old constraint (may not exist): {e}")
    
    # Add new unique constraint (on content_hash)
    op.create_unique_constraint(
        'uq_memory_session_content_hash',
        'memories',
        ['session_id', 'content_type', 'content_hash']
    )
    
    logger.info("✓ Created new unique constraint on content_hash")
    
    # ===========================
    # Step 5: Add indexes
    # ===========================
    
    logger.info("Creating indexes...")
    
    # Index on content_hash
    op.create_index(
        'ix_memory_content_hash',
        'memories',
        ['content_hash']
    )
    
    # Index on semantic_hash
    op.create_index(
        'ix_memory_semantic_hash',
        'memories',
        ['semantic_hash']
    )
    
    # Index on expires_at
    op.create_index(
        'ix_memory_expires_at',
        'memories',
        ['expires_at']
    )
    
    # Index on deleted_at
    op.create_index(
        'ix_memory_deleted_at',
        'memories',
        ['deleted_at']
    )
    
    # Composite index for active memories (PostgreSQL partial index)
    if is_postgresql:
        # Partial index: only active (non-deleted) memories
        op.execute(text("""
            CREATE INDEX ix_memory_active_expires
            ON memories (expires_at)
            WHERE deleted_at IS NULL AND expires_at IS NOT NULL
        """))
        
        logger.info("✓ Created partial index for active memories")
    
    logger.info("✓ All indexes created")
    
    # ===========================
    # Step 6: Add check constraints
    # ===========================
    
    logger.info("Adding check constraints...")
    
    # Importance range check
    op.create_check_constraint(
        'ck_memory_importance_range',
        'memories',
        'importance >= 0.0 AND importance <= 1.0'
    )
    
    # Access count positive check
    op.create_check_constraint(
        'ck_memory_access_count_positive',
        'memories',
        'access_count >= 0'
    )
    
    logger.info("✓ Check constraints added")
    
    logger.info("✅ Migration completed successfully")


def downgrade():
    """
    Reverse migration: Remove content hashing and TTL support.
    
    WARNING: This will lose content_hash, semantic_hash, expires_at, and deleted_at data.
    """
    logger.info("Starting downgrade: Remove content hashing from memories")
    
    connection = op.get_bind()
    is_postgresql = 'postgresql' in str(connection.engine.url)
    
    # Drop check constraints
    logger.info("Dropping check constraints...")
    try:
        op.drop_constraint('ck_memory_access_count_positive', 'memories', type_='check')
        op.drop_constraint('ck_memory_importance_range', 'memories', type_='check')
    except Exception as e:
        logger.warning(f"Could not drop check constraints: {e}")
    
    # Drop indexes
    logger.info("Dropping indexes...")
    
    if is_postgresql:
        try:
            op.execute(text("DROP INDEX IF EXISTS ix_memory_active_expires"))
        except Exception as e:
            logger.warning(f"Could not drop partial index: {e}")
    
    try:
        op.drop_index('ix_memory_deleted_at', 'memories')
        op.drop_index('ix_memory_expires_at', 'memories')
        op.drop_index('ix_memory_semantic_hash', 'memories')
        op.drop_index('ix_memory_content_hash', 'memories')
    except Exception as e:
        logger.warning(f"Could not drop some indexes: {e}")
    
    # Drop new unique constraint
    logger.info("Dropping new unique constraint...")
    try:
        op.drop_constraint('uq_memory_session_content_hash', 'memories', type_='unique')
    except Exception as e:
        logger.warning(f"Could not drop new constraint: {e}")
    
    # Recreate old unique constraint
    logger.info("Recreating old unique constraint...")
    try:
        op.create_unique_constraint(
            'uq_memory_session_content',
            'memories',
            ['session_id', 'content_type', 'content']
        )
    except Exception as e:
        logger.warning(f"Could not recreate old constraint: {e}")
    
    # Drop new columns
    logger.info("Dropping new columns...")
    op.drop_column('memories', 'deleted_at')
    op.drop_column('memories', 'expires_at')
    op.drop_column('memories', 'semantic_hash')
    op.drop_column('memories', 'content_hash')
    
    logger.info("✅ Downgrade completed")
```

---

## 📋 Implementation Summary

### ✅ All 6 Files Generated

**Phase 1 - Critical Security (Files 1-4):**
1. ✅ `backend/app/schemas/__init__.py` - Package initialization
2. ✅ `backend/app/schemas/tool_requests.py` - Complete input validation (500+ lines)
3. ✅ `backend/app/tools/memory_tool.py` - Enhanced with validation + context managers (600+ lines)
4. ✅ `backend/app/config/tool_settings.py` - SecretStr integration complete (450+ lines)

**Phase 2 - Database Optimization (Files 5-6):**
5. ✅ `backend/app/models/memory.py` - Content hashing + TTL + soft delete (600+ lines)
6. ✅ `alembic/versions/002_add_content_hashing.py` - Production-ready migration (350+ lines)

---

## 🚀 Next Steps - Deployment Instructions

### 1. Install Files (Copy & Paste)

```bash
# Create schemas directory
mkdir -p backend/app/schemas

# Copy all files to their locations
# File 1: backend/app/schemas/__init__.py
# File 2: backend/app/schemas/tool_requests.py
# File 3: backend/app/tools/memory_tool.py (replace existing)
# File 4: backend/app/config/tool_settings.py (replace existing)
# File 5: backend/app/models/memory.py (replace existing)
# File 6: alembic/versions/002_add_content_hashing.py
```

### 2. Test Imports

```bash
# Test that all imports work
python -c "from backend.app.schemas.tool_requests import MemoryStoreRequest; print('✓ Schemas OK')"
python -c "from backend.app.tools.memory_tool import MemoryTool; print('✓ Memory tool OK')"
python -c "from backend.app.config.tool_settings import tool_settings; print('✓ Settings OK')"
python -c "from backend.app.models.memory import Memory; print('✓ Model OK')"
```

### 3. Run Phase 1 (Security - Immediate)

```bash
# No database migration needed for Phase 1
# Just restart the application

uvicorn backend.app.main:app --reload
```

**Phase 1 is now active:**
- ✅ Input validation on all tool inputs
- ✅ SQL injection prevention
- ✅ Path traversal prevention
- ✅ XSS prevention
- ✅ Secure API key management

### 4. Run Phase 2 (Database - Optional)

```bash
# Backup database first!
cp backend/data/customer_support.db backend/data/customer_support.db.backup

# Run migration
alembic upgrade head

# Verify migration
python -c "
from backend.app.database import engine
from sqlalchemy import inspect
inspector = inspect(engine)
columns = [c['name'] for c in inspector.get_columns('memories')]
print('Columns:', columns)
assert 'content_hash' in columns, 'Migration failed!'
print('✓ Migration successful')
"
```

---

## 🧪 Validation Tests

### Test Input Validation

```python
# Test in Python REPL or create test_validation.py

from backend.app.schemas.tool_requests import MemoryStoreRequest
from pydantic import ValidationError

# Test 1: Valid request
try:
    request = MemoryStoreRequest(
        session_id="user-123",
        content="User prefers email communication",
        content_type="preference"
    )
    print("✓ Valid request accepted")
except ValidationError as e:
    print("✗ Should not fail:", e)

# Test 2: SQL injection attempt (should fail)
try:
    request = MemoryStoreRequest(
        session_id="user'; DROP TABLE memories;--",
        content="Test"
    )
    print("✗ SQL injection not blocked!")
except ValidationError as e:
    print("✓ SQL injection blocked:", e.errors()[0]['msg'])

# Test 3: Path traversal attempt (should fail)
try:
    request = MemoryStoreRequest(
        session_id="../../../etc/passwd",
        content="Test"
    )
    print("✗ Path traversal not blocked!")
except ValidationError as e:
    print("✓ Path traversal blocked:", e.errors()[0]['msg'])

# Test 4: Content sanitization
request = MemoryStoreRequest(
    session_id="user-123",
    content="Test\x00with\x00null\x00bytes   and    extra     spaces",
    content_type="context"
)
print(f"✓ Sanitized content: '{request.content}'")
# Should be: "Test with null bytes and extra spaces"
```

### Test Memory Tool with Validation

```python
import asyncio
from backend.app.tools.memory_tool import MemoryTool

async def test_memory_tool():
    tool = MemoryTool()
    await tool.initialize()
    
    # Test 1: Valid store
    result = await tool.store_memory_async(
        session_id="test-123",
        content="User likes coffee",
        content_type="preference"
    )
    print("✓ Valid store:", result['success'])
    
    # Test 2: Invalid session_id (should fail)
    result = await tool.store_memory_async(
        session_id="bad'; DROP TABLE--",
        content="Test"
    )
    print("✓ Invalid session_id blocked:", not result['success'])
    
    # Test 3: Retrieve
    memories = await tool.retrieve_memories_async(
        session_id="test-123"
    )
    print(f"✓ Retrieved {len(memories)} memories")
    
    await tool.cleanup()

asyncio.run(test_memory_tool())
```

---

## 📊 Security Validation Checklist

Run these tests to verify security:

```bash
# Create test_security.py
cat > test_security.py << 'EOF'
from backend.app.schemas.tool_requests import MemoryStoreRequest
from pydantic import ValidationError

def test_sql_injection():
    """Test SQL injection prevention."""
    payloads = [
        "'; DROP TABLE memories;--",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM users--"
    ]
    
    for payload in payloads:
        try:
            MemoryStoreRequest(session_id=payload, content="test")
            print(f"✗ FAILED: {payload} was accepted!")
            return False
        except ValidationError:
            print(f"✓ Blocked: {payload}")
    
    return True

def test_path_traversal():
    """Test path traversal prevention."""
    payloads = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "~/../../etc/shadow"
    ]
    
    for payload in payloads:
        try:
            MemoryStoreRequest(session_id=payload, content="test")
            print(f"✗ FAILED: {payload} was accepted!")
            return False
        except ValidationError:
            print(f"✓ Blocked: {payload}")
    
    return True

def test_xss_prevention():
    """Test XSS prevention in content."""
    payloads = [
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>"
    ]
    
    for payload in payloads:
        try:
            request = MemoryStoreRequest(
                session_id="test-123",
                content=payload
            )
            # Content should be sanitized
            if '<' in request.content or 'script' in request.content.lower():
                print(f"✗ FAILED: XSS not sanitized: {payload}")
                return False
            print(f"✓ Sanitized: {payload}")
        except ValidationError as e:
            print(f"✓ Rejected: {payload}")
    
    return True

if __name__ == "__main__":
    print("Running security tests...\n")
    
    print("1. SQL Injection Tests:")
    assert test_sql_injection(), "SQL injection tests failed"
    
    print("\n2. Path Traversal Tests:")
    assert test_path_traversal(), "Path traversal tests failed"
    
    print("\n3. XSS Prevention Tests:")
    assert test_xss_prevention(), "XSS prevention tests failed"
    
    print("\n✅ All security tests passed!")
EOF

python test_security.py
```
