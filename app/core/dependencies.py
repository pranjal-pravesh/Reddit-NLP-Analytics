from typing import Optional

from fastapi import Depends, HTTPException, Query, status

from app.core.config import settings


def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(
        settings.DEFAULT_PAGE_SIZE,
        ge=1, 
        le=settings.MAX_PAGE_SIZE,
        description=f"Items per page (max: {settings.MAX_PAGE_SIZE})"
    ),
):
    """
    Common pagination parameters as a dependency.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        Tuple of (skip, limit) for pagination
    """
    skip = (page - 1) * page_size
    return {"skip": skip, "limit": page_size, "page": page, "page_size": page_size}


def get_reddit_timeframe(
    timeframe: str = Query(
        "week",
        description="Time period filter",
        regex="^(hour|day|week|month|year|all)$"
    )
):
    """
    Validate and return Reddit timeframe parameter.
    
    Args:
        timeframe: Reddit timeframe string (hour, day, week, month, year, all)
        
    Returns:
        Validated timeframe string
    """
    valid_timeframes = ["hour", "day", "week", "month", "year", "all"]
    if timeframe not in valid_timeframes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
        )
    return timeframe


def get_reddit_sort(
    sort: str = Query(
        "hot",
        description="Sort method for Reddit results",
        regex="^(hot|new|top|rising|controversial|relevance|comments)$"
    )
):
    """
    Validate and return Reddit sort parameter.
    
    Args:
        sort: Reddit sort method (hot, new, top, rising, controversial, relevance, comments)
        
    Returns:
        Validated sort string
    """
    valid_sorts = ["hot", "new", "top", "rising", "controversial", "relevance", "comments"]
    if sort not in valid_sorts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sort method. Must be one of: {', '.join(valid_sorts)}"
        )
    return sort 