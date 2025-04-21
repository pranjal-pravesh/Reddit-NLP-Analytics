import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Tuple

import httpx
import praw
from praw.models import Submission, Subreddit

from app.core.config import settings
from app.utils.caching import redis_cache
from app.utils.logging_config import logger


class RedditClient:
    """
    Client for interacting with Reddit API, with support for:
    - Authenticated access via PRAW (if credentials provided)
    - Unauthenticated access via public API (fallback)
    """
    
    def __init__(self):
        self.authenticated = False
        self.reddit = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Try to initialize PRAW if credentials are provided
        if settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET:
            try:
                self.reddit = praw.Reddit(
                    client_id=settings.REDDIT_CLIENT_ID,
                    client_secret=settings.REDDIT_CLIENT_SECRET,
                    user_agent=settings.REDDIT_USER_AGENT
                )
                self.authenticated = True
                logger.info("Initialized authenticated Reddit client with PRAW")
            except Exception as e:
                logger.warning(f"Failed to initialize PRAW: {str(e)}")
                self.authenticated = False
        else:
            logger.info("No Reddit credentials provided, using unauthenticated access")
    
    async def close(self):
        """Clean up resources on shutdown"""
        await self.http_client.aclose()
    
    @redis_cache("reddit:search", ttl=1800)  # 30 minutes cache
    async def search(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        timeframe: str = "week",
        limit: int = 25,
        skip: int = 0
    ) -> Dict[str, Any]:
        """
        Search Reddit for posts matching the query.
        
        Args:
            query: Search query string
            subreddit: Optional subreddit to search in
            sort: Sort method (relevance, hot, new, top, etc.)
            timeframe: Time period (hour, day, week, month, year, all)
            limit: Maximum number of results to return
            skip: Number of results to skip (for pagination)
        
        Returns:
            Dict with search results and metadata
        """
        if self.authenticated:
            return await self._search_authenticated(
                query, subreddit, sort, timeframe, limit, skip
            )
        else:
            return await self._search_unauthenticated(
                query, subreddit, sort, timeframe, limit, skip
            )
    
    @redis_cache("reddit:subreddit", ttl=3600)  # 1 hour cache
    async def get_subreddit_info(self, subreddit_name: str) -> Dict[str, Any]:
        """
        Get information about a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
        
        Returns:
            Dict with subreddit information
        """
        if self.authenticated:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                # Force fetch to validate subreddit exists
                display_name = subreddit.display_name
                
                return {
                    "display_name": subreddit.display_name,
                    "title": subreddit.title,
                    "description": subreddit.public_description,
                    "subscribers": subreddit.subscribers,
                    "created_utc": subreddit.created_utc,
                    "over18": subreddit.over18,
                    "url": subreddit.url
                }
            except Exception as e:
                logger.error(f"Error fetching subreddit {subreddit_name}: {str(e)}")
                raise
        else:
            url = f"https://www.reddit.com/r/{subreddit_name}/about.json"
            async with self.http_client.stream("GET", url) as response:
                if response.status_code != 200:
                    logger.error(f"Error from Reddit API: {response.status_code}")
                    raise Exception(f"Reddit API error: {response.status_code}")
                
                data = await response.json()
                subreddit_data = data.get("data", {})
                
                return {
                    "display_name": subreddit_data.get("display_name"),
                    "title": subreddit_data.get("title"),
                    "description": subreddit_data.get("public_description"),
                    "subscribers": subreddit_data.get("subscribers"),
                    "created_utc": subreddit_data.get("created_utc"),
                    "over18": subreddit_data.get("over18"),
                    "url": subreddit_data.get("url")
                }

    @redis_cache("reddit:post", ttl=3600)  # 1 hour cache
    async def get_post(self, post_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific Reddit post.
        
        Args:
            post_id: Reddit post ID
        
        Returns:
            Dict with post information
        """
        if self.authenticated:
            try:
                submission = self.reddit.submission(id=post_id)
                
                # Convert to dict
                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "author": submission.author.name if submission.author else "[deleted]",
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "permalink": submission.permalink,
                    "url": submission.url,
                    "is_self": submission.is_self,
                    "selftext": submission.selftext,
                    "subreddit": submission.subreddit.display_name
                }
                
                return post_data
            except Exception as e:
                logger.error(f"Error fetching post {post_id}: {str(e)}")
                raise
        else:
            url = f"https://www.reddit.com/comments/{post_id}.json"
            async with self.http_client.stream("GET", url) as response:
                if response.status_code != 200:
                    logger.error(f"Error from Reddit API: {response.status_code}")
                    raise Exception(f"Reddit API error: {response.status_code}")
                
                data = await response.json()
                post_data = data[0]["data"]["children"][0]["data"]
                
                return {
                    "id": post_data.get("id"),
                    "title": post_data.get("title"),
                    "author": post_data.get("author", "[deleted]"),
                    "created_utc": post_data.get("created_utc"),
                    "score": post_data.get("score"),
                    "upvote_ratio": post_data.get("upvote_ratio"),
                    "num_comments": post_data.get("num_comments"),
                    "permalink": post_data.get("permalink"),
                    "url": post_data.get("url"),
                    "is_self": post_data.get("is_self"),
                    "selftext": post_data.get("selftext"),
                    "subreddit": post_data.get("subreddit")
                }
    
    async def _search_authenticated(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        timeframe: str = "week",
        limit: int = 25,
        skip: int = 0
    ) -> Dict[str, Any]:
        """Perform authenticated search using PRAW"""
        total_limit = limit + skip
        results = []
        
        try:
            # Get subreddit instance or use None for all Reddit
            subreddit_obj = None
            if subreddit:
                subreddit_obj = self.reddit.subreddit(subreddit)
            
            # Perform search
            if subreddit_obj:
                search_results = subreddit_obj.search(
                    query, sort=sort, time_filter=timeframe, limit=total_limit
                )
            else:
                search_results = self.reddit.subreddit("all").search(
                    query, sort=sort, time_filter=timeframe, limit=total_limit
                )
            
            # Process results (need to use a list since we can't reuse the iterator)
            all_results = list(search_results)
            
            # Apply pagination
            paginated_results = all_results[skip:skip+limit]
            
            for post in paginated_results:
                results.append(self._format_post(post))
            
            return {
                "posts": results,
                "metadata": {
                    "count": len(results),
                    "query": query,
                    "subreddit": subreddit,
                    "sort": sort,
                    "timeframe": timeframe
                }
            }
        except Exception as e:
            logger.error(f"Error in authenticated search: {str(e)}")
            # Fall back to unauthenticated search
            return await self._search_unauthenticated(
                query, subreddit, sort, timeframe, limit, skip
            )
    
    async def _search_unauthenticated(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        timeframe: str = "week",
        limit: int = 25,
        skip: int = 0
    ) -> Dict[str, Any]:
        """Perform unauthenticated search using public Reddit API"""
        # Construct the search URL
        base_url = "https://www.reddit.com"
        
        if subreddit:
            url = f"{base_url}/r/{subreddit}/search.json"
        else:
            url = f"{base_url}/search.json"
        
        params = {
            "q": query,
            "sort": sort,
            "t": timeframe,
            "limit": limit,
            "count": skip,
            "raw_json": 1
        }
        
        try:
            async with self.http_client.stream("GET", url, params=params) as response:
                if response.status_code != 200:
                    logger.error(f"Error from Reddit API: {response.status_code}")
                    raise Exception(f"Reddit API error: {response.status_code}")
                
                data = await response.json()
                posts = data.get("data", {}).get("children", [])
                
                results = []
                for post in posts:
                    post_data = post.get("data", {})
                    results.append({
                        "id": post_data.get("id"),
                        "title": post_data.get("title"),
                        "author": post_data.get("author", "[deleted]"),
                        "created_utc": post_data.get("created_utc"),
                        "score": post_data.get("score"),
                        "upvote_ratio": post_data.get("upvote_ratio"),
                        "num_comments": post_data.get("num_comments"),
                        "permalink": post_data.get("permalink"),
                        "url": post_data.get("url"),
                        "is_self": post_data.get("is_self"),
                        "selftext": post_data.get("selftext"),
                        "subreddit": post_data.get("subreddit")
                    })
                
                return {
                    "posts": results,
                    "metadata": {
                        "count": len(results),
                        "query": query,
                        "subreddit": subreddit,
                        "sort": sort,
                        "timeframe": timeframe
                    }
                }
        
        except Exception as e:
            logger.error(f"Error in unauthenticated search: {str(e)}")
            raise
    
    def _format_post(self, post: Submission) -> Dict[str, Any]:
        """Convert PRAW Submission object to dict"""
        return {
            "id": post.id,
            "title": post.title,
            "author": post.author.name if post.author else "[deleted]",
            "created_utc": post.created_utc,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "permalink": post.permalink,
            "url": post.url,
            "is_self": post.is_self,
            "selftext": post.selftext,
            "subreddit": post.subreddit.display_name
        }

    async def get_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        timeframe: str = "week",
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Get posts from a specific subreddit.
        
        Args:
            subreddit: Subreddit name
            sort: Sort method (hot, new, top, rising, controversial)
            timeframe: Time period (hour, day, week, month, year, all)
            limit: Maximum number of results to return
        
        Returns:
            Dict with posts and metadata
        """
        if self.authenticated:
            try:
                subreddit_obj = self.reddit.subreddit(subreddit)
                # Force fetch to validate subreddit exists
                display_name = subreddit_obj.display_name
                
                # Get posts based on sort method
                if sort == "hot":
                    posts = subreddit_obj.hot(limit=limit)
                elif sort == "new":
                    posts = subreddit_obj.new(limit=limit)
                elif sort == "top":
                    posts = subreddit_obj.top(time_filter=timeframe, limit=limit)
                elif sort == "rising":
                    posts = subreddit_obj.rising(limit=limit)
                elif sort == "controversial":
                    posts = subreddit_obj.controversial(time_filter=timeframe, limit=limit)
                else:
                    raise ValueError(f"Invalid sort method: {sort}")
                
                results = []
                for post in posts:
                    results.append(self._format_post(post))
                
                return {
                    "posts": results,
                    "metadata": {
                        "count": len(results),
                        "subreddit": subreddit,
                        "sort": sort,
                        "timeframe": timeframe
                    }
                }
            except Exception as e:
                logger.error(f"Error fetching subreddit posts: {str(e)}")
                raise
        else:
            # Construct the URL for unauthenticated access
            base_url = f"https://www.reddit.com/r/{subreddit}"
            url = f"{base_url}/{sort}.json"
            
            params = {
                "limit": limit,
                "t": timeframe,
                "raw_json": 1
            }
            
            try:
                async with self.http_client.stream("GET", url, params=params) as response:
                    if response.status_code != 200:
                        logger.error(f"Error from Reddit API: {response.status_code}")
                        raise Exception(f"Reddit API error: {response.status_code}")
                    
                    data = await response.json()
                    posts = data.get("data", {}).get("children", [])
                    
                    results = []
                    for post in posts:
                        post_data = post.get("data", {})
                        results.append({
                            "id": post_data.get("id"),
                            "title": post_data.get("title"),
                            "author": post_data.get("author", "[deleted]"),
                            "created_utc": post_data.get("created_utc"),
                            "score": post_data.get("score"),
                            "upvote_ratio": post_data.get("upvote_ratio"),
                            "num_comments": post_data.get("num_comments"),
                            "permalink": post_data.get("permalink"),
                            "url": post_data.get("url"),
                            "is_self": post_data.get("is_self"),
                            "selftext": post_data.get("selftext"),
                            "subreddit": post_data.get("subreddit")
                        })
                    
                    return {
                        "posts": results,
                        "metadata": {
                            "count": len(results),
                            "subreddit": subreddit,
                            "sort": sort,
                            "timeframe": timeframe
                        }
                    }
            except Exception as e:
                logger.error(f"Error in unauthenticated subreddit posts fetch: {str(e)}")
                raise

    async def get_user_posts(
        self,
        username: str,
        sort: str = "new",
        timeframe: str = "all",
        limit: int = 25
    ) -> Dict[str, Any]:
        """
        Get posts from a specific Reddit user.
        
        Args:
            username: Reddit username
            sort: Sort method (hot, new, top, controversial)
            timeframe: Time period (hour, day, week, month, year, all)
            limit: Maximum number of results to return
        
        Returns:
            Dict with posts and metadata
        """
        if self.authenticated:
            try:
                user = self.reddit.redditor(username)
                # Force fetch to validate user exists
                name = user.name
                
                # Get posts based on sort method
                if sort == "hot":
                    posts = user.submissions.hot(limit=limit)
                elif sort == "new":
                    posts = user.submissions.new(limit=limit)
                elif sort == "top":
                    posts = user.submissions.top(time_filter=timeframe, limit=limit)
                elif sort == "controversial":
                    posts = user.submissions.controversial(time_filter=timeframe, limit=limit)
                else:
                    raise ValueError(f"Invalid sort method: {sort}")
                
                results = []
                for post in posts:
                    results.append(self._format_post(post))
                
                return {
                    "posts": results,
                    "metadata": {
                        "count": len(results),
                        "username": username,
                        "sort": sort,
                        "timeframe": timeframe
                    }
                }
            except Exception as e:
                logger.error(f"Error fetching user posts: {str(e)}")
                raise
        else:
            # Construct the URL for unauthenticated access
            base_url = f"https://www.reddit.com/user/{username}"
            url = f"{base_url}/submitted.json"
            
            params = {
                "limit": limit,
                "sort": sort,
                "t": timeframe,
                "raw_json": 1
            }
            
            try:
                async with self.http_client.stream("GET", url, params=params) as response:
                    if response.status_code != 200:
                        logger.error(f"Error from Reddit API: {response.status_code}")
                        raise Exception(f"Reddit API error: {response.status_code}")
                    
                    data = await response.json()
                    posts = data.get("data", {}).get("children", [])
                    
                    results = []
                    for post in posts:
                        post_data = post.get("data", {})
                        results.append({
                            "id": post_data.get("id"),
                            "title": post_data.get("title"),
                            "author": post_data.get("author", "[deleted]"),
                            "created_utc": post_data.get("created_utc"),
                            "score": post_data.get("score"),
                            "upvote_ratio": post_data.get("upvote_ratio"),
                            "num_comments": post_data.get("num_comments"),
                            "permalink": post_data.get("permalink"),
                            "url": post_data.get("url"),
                            "is_self": post_data.get("is_self"),
                            "selftext": post_data.get("selftext"),
                            "subreddit": post_data.get("subreddit")
                        })
                    
                    return {
                        "posts": results,
                        "metadata": {
                            "count": len(results),
                            "username": username,
                            "sort": sort,
                            "timeframe": timeframe
                        }
                    }
            except Exception as e:
                logger.error(f"Error in unauthenticated user posts fetch: {str(e)}")
                raise


# Create a global instance
reddit_client = RedditClient() 