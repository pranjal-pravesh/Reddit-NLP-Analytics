import functools
import hashlib
import json
import pickle
from typing import Any, Callable, Optional, TypeVar, Union, Dict, Tuple, List
import time

import redis
from app.core.config import settings
from app.utils.logging_config import logger

T = TypeVar("T")

# Initialize Redis connection if enabled
_redis_client = None

if settings.ENABLE_REDIS_CACHE:
    try:
        _redis_client = redis.from_url(settings.REDIS_URL)
        _redis_client.ping()  # Test connection
        logger.info(f"Redis cache connected at {settings.REDIS_URL}")
    except redis.exceptions.ConnectionError:
        logger.warning(f"Failed to connect to Redis at {settings.REDIS_URL}")
        _redis_client = None


def _create_key(prefix: str, args: Tuple, kwargs: Dict) -> str:
    """Create a unique cache key based on function arguments"""
    # Create a string representation of args and kwargs
    key_dict = {
        "args": args,
        "kwargs": kwargs
    }
    key_str = json.dumps(key_dict, sort_keys=True, default=str)
    
    # Hash the string to create a fixed-length key
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    return f"{prefix}:{key_hash}"


def redis_cache(prefix: str, ttl: Optional[int] = None):
    """
    Redis caching decorator with TTL support.
    Uses function-specific prefix and arguments to create unique keys.
    
    Args:
        prefix: Prefix for the cache key
        ttl: Time-to-live for the cache in seconds, None for no expiry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            # If Redis is not enabled, just call the function
            if not settings.ENABLE_REDIS_CACHE or _redis_client is None:
                return await func(*args, **kwargs)
            
            # Create a unique key
            key = _create_key(prefix, args, kwargs)
            
            try:
                # Try to get from cache
                cached = _redis_client.get(key)
                if cached:
                    logger.debug(f"Cache hit: {key}")
                    return pickle.loads(cached)
                
                # If not in cache, call the function
                result = await func(*args, **kwargs)
                
                # Store result in cache
                serialized_result = pickle.dumps(result)
                expiry = ttl if ttl is not None else settings.CACHE_TTL
                _redis_client.setex(key, expiry, serialized_result)
                logger.debug(f"Cache set: {key} (TTL: {expiry}s)")
                
                return result
            except Exception as e:
                logger.warning(f"Redis cache error: {str(e)}")
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            # If Redis is not enabled, just call the function
            if not settings.ENABLE_REDIS_CACHE or _redis_client is None:
                return func(*args, **kwargs)
            
            # Create a unique key
            key = _create_key(prefix, args, kwargs)
            
            try:
                # Try to get from cache
                cached = _redis_client.get(key)
                if cached:
                    logger.debug(f"Cache hit: {key}")
                    return pickle.loads(cached)
                
                # If not in cache, call the function
                result = func(*args, **kwargs)
                
                # Store result in cache
                serialized_result = pickle.dumps(result)
                expiry = ttl if ttl is not None else settings.CACHE_TTL
                _redis_client.setex(key, expiry, serialized_result)
                logger.debug(f"Cache set: {key} (TTL: {expiry}s)")
                
                return result
            except Exception as e:
                logger.warning(f"Redis cache error: {str(e)}")
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio_callable(func) else sync_wrapper
    
    return decorator


def memory_cache(maxsize: int = 128, ttl: Optional[int] = 3600):
    """
    In-memory LRU cache with optional TTL.
    
    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds, None for no expiry
    """
    cache_dict = {}
    cache_times = {}
    
    def make_hashable(obj):
        """Convert unhashable types to hashable types for caching"""
        if isinstance(obj, list):
            return tuple(make_hashable(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, set):
            return tuple(sorted(make_hashable(item) for item in obj))
        return obj
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cached_func = functools.lru_cache(maxsize=maxsize)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                # Try to create a hashable key by converting unhashable types
                hashable_args = tuple(make_hashable(arg) for arg in args)
                hashable_kwargs = tuple(sorted((k, make_hashable(v)) for k, v in kwargs.items()))
                key = (hashable_args, hashable_kwargs)
                
                # Check TTL if enabled
                if ttl is not None and key in cache_times:
                    current_time = time.time()
                    if current_time - cache_times[key] > ttl:
                        # Expired, remove from dictionary
                        cache_dict.pop(key, None)
                        cache_times.pop(key, None)
                
                # Get or compute result
                if key not in cache_dict:
                    result = func(*args, **kwargs)
                    cache_dict[key] = result
                    if ttl is not None:
                        cache_times[key] = time.time()
                    return result
                
                return cache_dict[key]
            except Exception as e:
                # If caching fails for any reason, just call the function directly
                logger.warning(f"Memory cache error: {str(e)}")
                return func(*args, **kwargs)
            
        return wrapper
    
    return decorator


def asyncio_callable(obj: Any) -> bool:
    """Check if an object is an async callable"""
    return hasattr(obj, "__await__") or hasattr(obj, "__aenter__") 