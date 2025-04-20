from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status

from app.api.v1.schemas.reddit import (
    RedditSearchParams,
    RedditSearchResponse,
    SubredditResponse,
)
from app.core.dependencies import get_pagination_params, get_reddit_sort, get_reddit_timeframe
from app.services.reddit_client import reddit_client

router = APIRouter()


@router.get("/search", response_model=RedditSearchResponse)
async def search_reddit(
    query: str = Query(..., min_length=1, description="Search query"),
    subreddit: Optional[str] = Query(None, description="Specific subreddit to search in"),
    sort: str = Depends(get_reddit_sort),
    timeframe: str = Depends(get_reddit_timeframe),
    pagination: Dict[str, int] = Depends(get_pagination_params),
):
    """
    Search Reddit for posts matching the query.
    
    Supports filtering by subreddit, sorting, and time period.
    Results are paginated.
    """
    try:
        result = await reddit_client.search(
            query=query,
            subreddit=subreddit,
            sort=sort,
            timeframe=timeframe,
            limit=pagination["limit"],
            skip=pagination["skip"]
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching Reddit: {str(e)}"
        )


@router.get("/subreddit/{subreddit_name}", response_model=SubredditResponse)
async def get_subreddit(
    subreddit_name: str = Path(..., min_length=1, description="Subreddit name")
):
    """
    Get information about a specific subreddit.
    """
    try:
        subreddit_info = await reddit_client.get_subreddit_info(subreddit_name)
        return {"subreddit": subreddit_info}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching subreddit information: {str(e)}"
        )


@router.get("/post/{post_id}")
async def get_post(
    post_id: str = Path(..., min_length=1, description="Reddit post ID")
):
    """
    Get detailed information about a specific Reddit post.
    """
    try:
        post_data = await reddit_client.get_post(post_id)
        return post_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching post: {str(e)}"
        ) 