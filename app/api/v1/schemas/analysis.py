from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""
    text: str = Field(..., min_length=1, description="Text to analyze")


class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis results"""
    label: str
    score: float
    positive: Optional[float] = None
    negative: Optional[float] = None
    neutral: Optional[float] = None


class KeywordExtractionRequest(BaseModel):
    """Request model for keyword extraction"""
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")
    method: str = Field("tfidf", description="Keyword extraction method (tfidf or textrank)")
    num_keywords: int = Field(10, ge=1, le=50, description="Number of keywords to extract")
    
    @validator("method")
    def validate_method(cls, v):
        valid_methods = ["tfidf", "textrank"]
        if v not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of: {', '.join(valid_methods)}")
        return v


class KeywordResult(BaseModel):
    """Keyword extraction result for a single text"""
    keywords: List[str]
    scores: List[float]


class KeywordExtractionResponse(BaseModel):
    """Response model for keyword extraction results"""
    results: List[KeywordResult]


class TopicModelingRequest(BaseModel):
    """Request model for topic modeling"""
    texts: List[str] = Field(..., min_items=5, description="List of texts to analyze")
    num_topics: int = Field(5, ge=2, le=20, description="Number of topics to identify")
    method: str = Field("lda", description="Topic modeling method (lda or bertopic)")
    
    @validator("method")
    def validate_method(cls, v):
        valid_methods = ["lda", "bertopic"]
        if v not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of: {', '.join(valid_methods)}")
        return v


class Topic(BaseModel):
    """Topic model with keywords and probabilities"""
    id: int
    words: List[str]
    probabilities: List[float]


class DocumentTopics(BaseModel):
    """Topic distribution for a document"""
    topic_ids: List[int]
    probabilities: List[float]


class TopicModelingResponse(BaseModel):
    """Response model for topic modeling results"""
    topics: List[Topic]
    topic_distribution: List[DocumentTopics]


class TimeHistogramRequest(BaseModel):
    """Request model for time-based histogram"""
    timestamps: List[float] = Field(..., min_items=1, description="List of UTC timestamps")
    interval: str = Field("day", description="Time interval (hour, day, week, month)")
    
    @validator("interval")
    def validate_interval(cls, v):
        valid_intervals = ["hour", "day", "week", "month"]
        if v not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {', '.join(valid_intervals)}")
        return v


class TimeHistogramResponse(BaseModel):
    """Response model for time-based histogram"""
    counts: List[int]
    labels: List[str]
    plot: Optional[str] = None  # Base64 encoded image


class BatchAnalysisRequest(BaseModel):
    """Request model for batch processing of texts"""
    operation: str = Field(..., description="Operation to perform (sentiment, keywords, topics)")
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")
    params: Optional[Dict[str, Any]] = Field(None, description="Operation-specific parameters")
    
    @validator("operation")
    def validate_operation(cls, v):
        valid_operations = ["sentiment", "keywords", "topics"]
        if v not in valid_operations:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_operations)}")
        return v


class BatchAnalysisResponse(BaseModel):
    """Response model for batch processing results"""
    results: List[Dict[str, Any]]
    operation: str


# LLM Integration Schemas
class LLMRequest(BaseModel):
    """Base request model for LLM operations"""
    text: str = Field(..., min_length=1, description="Text to process")
    provider: Optional[str] = Field(None, description="Specific LLM provider to use")
    
    @validator("provider")
    def validate_provider(cls, v):
        if v is not None:
            valid_providers = ["openai", "anthropic", "google"]
            if v not in valid_providers:
                raise ValueError(f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
        return v


class LLMSummaryRequest(LLMRequest):
    """Request model for LLM text summarization"""
    max_length: Optional[int] = Field(2000, ge=100, le=10000, description="Maximum text length to process")


class LLMSummaryResponse(BaseModel):
    """Response model for LLM summarization"""
    text: str
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None


class LLMKeywordClusterRequest(BaseModel):
    """Request model for LLM keyword clustering"""
    keywords: List[str] = Field(..., min_items=5, description="Keywords to cluster")
    provider: Optional[str] = Field(None, description="Specific LLM provider to use")
    
    @validator("provider")
    def validate_provider(cls, v):
        if v is not None:
            valid_providers = ["openai", "anthropic", "google"]
            if v not in valid_providers:
                raise ValueError(f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
        return v


class LLMKeywordClusterResponse(BaseModel):
    """Response model for LLM keyword clustering"""
    text: str
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None 