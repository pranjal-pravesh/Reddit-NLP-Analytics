from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, Body

from app.api.v1.schemas.reddit import (
    RedditSearchParams,
    RedditSearchResponse,
    SubredditResponse,
    SubredditRequest,
    RedditSearchRequest,
    UserRequest,
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


@router.post("/search", response_model=RedditSearchResponse)
async def search_reddit_post(
    request: RedditSearchRequest = Body(...)
):
    """
    POST endpoint for searching Reddit posts matching a query.
    
    Accepts query, sort, and time_filter parameters in the request body.
    """
    try:
        result = await reddit_client.search(
            query=request.query,
            subreddit=None,
            sort=request.sort,
            timeframe=request.time_filter,
            limit=request.limit,
            skip=0
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


@router.post("/subreddit", response_model=RedditSearchResponse)
async def get_subreddit_posts(
    request: SubredditRequest = Body(...)
):
    """
    POST endpoint for fetching posts from a specific subreddit.
    
    Accepts subreddit name, sort, and time_filter parameters in the request body.
    """
    try:
        result = await reddit_client.get_subreddit_posts(
            subreddit=request.subreddit,
            sort=request.sort,
            timeframe=request.time_filter,
            limit=request.limit
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching subreddit posts: {str(e)}"
        )


@router.post("/user", response_model=RedditSearchResponse)
async def get_user_posts(
    request: UserRequest = Body(...)
):
    """
    POST endpoint for fetching posts from a specific Reddit user.
    
    Accepts username, sort, and time_filter parameters in the request body.
    """
    try:
        result = await reddit_client.get_user_posts(
            username=request.username,
            sort=request.sort,
            timeframe=request.time_filter,
            limit=request.limit
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user posts: {str(e)}"
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