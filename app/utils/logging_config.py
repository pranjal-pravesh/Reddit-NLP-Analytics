import logging
import sys
from pathlib import Path

from app.core.config import settings


def setup_logging():
    """Configure application logging based on settings"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    handlers = [
        logging.StreamHandler(sys.stdout),  # Console handler
    ]
    
    # Add file handler in non-development environments
    if settings.APP_ENV != "development":
        file_handler = logging.FileHandler(
            log_dir / f"reddit_analysis_{settings.APP_ENV}.log"
        )
        handlers.append(file_handler)
    
    # Configure logging format
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    
    # Set lower log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("praw").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Create logger
    logger = logging.getLogger("reddit_analysis")
    logger.setLevel(log_level)
    
    return logger


# Create global logger instance
logger = setup_logging() 