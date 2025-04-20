import os
from typing import Dict, List, Optional, Any, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed through environment variables and pydantic.
    """
    # App settings
    APP_ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Reddit API Settings
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "RedditAnalysisApp/1.0"
    
    # LLM Integration
    ENABLE_LLM_INTEGRATION: bool = False
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-instant-1"
    
    # Google
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_MODEL: str = "gemini-pro"
    
    # Caching settings
    ENABLE_REDIS_CACHE: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # seconds
    
    # API settings
    DEFAULT_PAGE_SIZE: int = 25
    MAX_PAGE_SIZE: int = 100
    
    @field_validator("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", mode="after")
    def check_reddit_credentials(cls, v, values):
        """Validate that Reddit credentials are provided together if any are set"""
        if v is not None and (values.data.get("REDDIT_CLIENT_ID") is None or values.data.get("REDDIT_CLIENT_SECRET") is None):
            print("Warning: Both REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET should be provided for authenticated access")
        return v
    
    @field_validator("ENABLE_LLM_INTEGRATION", mode="after")
    def check_llm_credentials(cls, v, values):
        """Check if LLM integration is enabled, then at least one provider needs credentials"""
        if v:
            has_openai = values.data.get("OPENAI_API_KEY") is not None
            has_anthropic = values.data.get("ANTHROPIC_API_KEY") is not None
            has_google = values.data.get("GOOGLE_API_KEY") is not None
            
            if not (has_openai or has_anthropic or has_google):
                print("Warning: LLM integration is enabled but no provider API keys are set")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Create global settings instance
settings = Settings() 