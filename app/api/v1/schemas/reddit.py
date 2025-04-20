from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class RedditSearchParams(BaseModel):
    """Parameters for Reddit search endpoint"""
    query: str = Field(..., min_length=1, description="Search query")
    subreddit: Optional[str] = Field(None, description="Specific subreddit to search in")
    sort: str = Field("relevance", description="Sort method (relevance, hot, new, top, rising, controversial)")
    timeframe: str = Field("week", description="Time period (hour, day, week, month, year, all)")
    
    @validator("sort")
    def validate_sort(cls, v):
        valid_sorts = ["relevance", "hot", "new", "top", "rising", "controversial"]
        if v not in valid_sorts:
            raise ValueError(f"Invalid sort. Must be one of: {', '.join(valid_sorts)}")
        return v
    
    @validator("timeframe")
    def validate_timeframe(cls, v):
        valid_timeframes = ["hour", "day", "week", "month", "year", "all"]
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}")
        return v


class RedditPost(BaseModel):
    """Reddit post model"""
    id: str
    title: str
    author: str
    created_utc: float
    score: int
    upvote_ratio: Optional[float] = None
    num_comments: int
    permalink: str
    url: str
    is_self: bool
    selftext: Optional[str] = None
    subreddit: str


class RedditSearchResponse(BaseModel):
    """Response model for Reddit search results"""
    posts: List[RedditPost]
    metadata: Dict[str, Any]


class SubredditInfo(BaseModel):
    """Subreddit information model"""
    display_name: str
    title: str
    description: Optional[str] = None
    subscribers: Optional[int] = None
    created_utc: Optional[float] = None
    over18: Optional[bool] = None
    url: Optional[str] = None


class SubredditResponse(BaseModel):
    """Response model for subreddit info"""
    subreddit: SubredditInfo 