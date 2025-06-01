import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.config import settings
from app.utils.logging_config import logger


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class BaseLLMClient(ABC):
    """Base abstract class for all LLM clients"""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text from the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dict with generation results
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.client = None
        
        if not self.api_key:
            logger.warning("OpenAI API key is not set")
            return
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with model: {self.model}")
        except ImportError:
            logger.error("Failed to import OpenAI package")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.client = None
    
    async def generate(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using OpenAI API"""
        if not self.client:
            return {"error": "OpenAI client not initialized", "text": ""}
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": self.model,
                "provider": LLMProvider.OPENAI.value,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error in OpenAI generation: {str(e)}")
            return {"error": str(e), "text": "", "provider": LLMProvider.OPENAI.value}
    
    async def close(self):
        """No resources to clean up for OpenAI"""
        pass


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API"""
    
    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        self.model = settings.ANTHROPIC_MODEL
        self.client = None
        
        if not self.api_key:
            logger.warning("Anthropic API key is not set")
            return
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            logger.info(f"Initialized Anthropic client with model: {self.model}")
        except ImportError:
            logger.error("Failed to import Anthropic package")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {str(e)}")
            self.client = None
    
    async def generate(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using Anthropic Claude API"""
        if not self.client:
            return {"error": "Anthropic client not initialized", "text": ""}
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "text": response.content[0].text,
                "model": self.model,
                "provider": LLMProvider.ANTHROPIC.value,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error in Anthropic generation: {str(e)}")
            return {"error": str(e), "text": "", "provider": LLMProvider.ANTHROPIC.value}
    
    async def close(self):
        """No resources to clean up for Anthropic"""
        pass


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini API"""
    
    def __init__(self):
        self.api_key = settings.GOOGLE_API_KEY
        self.model = settings.GOOGLE_MODEL
        self.client = None
        
        if not self.api_key:
            logger.warning("Google API key is not set")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
            logger.info(f"Initialized Google Gemini client with model: {self.model}")
        except ImportError:
            logger.error("Failed to import Google GenerativeAI package")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Google client: {str(e)}")
            self.client = None
    
    async def generate(
        self, 
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using Google Gemini API"""
        if not self.client:
            return {"error": "Google client not initialized", "text": ""}
        
        try:
            # Convert to async using run_in_executor
            loop = asyncio.get_event_loop()
            model = self.client.GenerativeModel(self.model)
            
            # Google API is synchronous, so we run it in a thread
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
            )
            
            return {
                "text": response.text,
                "model": self.model,
                "provider": LLMProvider.GOOGLE.value
            }
        except Exception as e:
            logger.error(f"Error in Google generation: {str(e)}")
            return {"error": str(e), "text": "", "provider": LLMProvider.GOOGLE.value}
    
    async def close(self):
        """No resources to clean up for Google"""
        pass


class LLMService:
    """
    Service for LLM operations with:
    - Provider-agnostic interface
    - Support for OpenAI, Anthropic Claude, and Google Gemini
    - Batch processing and rate limiting
    """
    
    def __init__(self):
        self.enabled = settings.ENABLE_LLM_INTEGRATION
        self.clients: Dict[LLMProvider, BaseLLMClient] = {}
        
        if not self.enabled:
            logger.info("LLM integration is disabled in settings")
            return
        
        # Initialize available clients
        if settings.OPENAI_API_KEY:
            self.clients[LLMProvider.OPENAI] = OpenAIClient()
        
        if settings.ANTHROPIC_API_KEY:
            self.clients[LLMProvider.ANTHROPIC] = AnthropicClient()
        
        if settings.GOOGLE_API_KEY:
            self.clients[LLMProvider.GOOGLE] = GoogleClient()
        
        if not self.clients:
            logger.warning("LLM integration is enabled but no provider API keys are set")
            self.enabled = False
    
    async def close(self):
        """Clean up resources for all clients"""
        for client in self.clients.values():
            await client.close()
    
    async def summarize_text(
        self, 
        text: str,
        provider: Optional[LLMProvider] = None,
        max_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate a concise summary of the provided text.
        
        Args:
            text: Text to summarize
            provider: Specific LLM provider to use (or None for default)
            max_length: Maximum length of text to process
            
        Returns:
            Dict with summary result
        """
        if not self.enabled or not self.clients:
            return {"summary": "", "error": "LLM integration is disabled or no providers available"}
        
        # Truncate text if necessary
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = f"""Please provide a concise summary of the following text. 
Focus on the main points and key information.

TEXT TO SUMMARIZE:
{text}

SUMMARY:"""
        
        return await self._generate(prompt, provider)
    
    async def extract_key_insights(
        self, 
        text: str,
        provider: Optional[LLMProvider] = None,
        max_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Extract key insights and topics from the provided text.
        
        Args:
            text: Text to analyze
            provider: Specific LLM provider to use (or None for default)
            max_length: Maximum length of text to process
            
        Returns:
            Dict with insights result
        """
        if not self.enabled or not self.clients:
            return {"insights": [], "error": "LLM integration is disabled or no providers available"}
        
        # Truncate text if necessary
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = f"""Please extract the 3-5 most important insights, topics, or themes from the following text.
Format them as a concise, bulleted list with a brief explanation for each.

TEXT TO ANALYZE:
{text}

KEY INSIGHTS:"""
        
        result = await self._generate(prompt, provider)
        
        # Try to parse the result into a list
        if "text" in result and result["text"]:
            # Very basic parsing of bullet points
            insights = []
            for line in result["text"].split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    insights.append(line[1:].strip())
            
            result["insights"] = insights
        
        return result
    
    async def cluster_keywords(
        self, 
        keywords: List[str],
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Cluster similar keywords and provide a label for each cluster.
        
        Args:
            keywords: List of keywords to cluster
            provider: Specific LLM provider to use (or None for default)
            
        Returns:
            Dict with clustering result
        """
        if not self.enabled or not self.clients:
            return {"clusters": [], "error": "LLM integration is disabled or no providers available"}
        
        keyword_text = ", ".join(keywords)
        
        prompt = f"""Given the following list of keywords, please group them into 3-5 meaningful clusters.
For each cluster, provide a descriptive label and list the keywords that belong to it.
Format your response as a clean, clear list of clusters with their respective keywords.

KEYWORDS:
{keyword_text}

CLUSTERS:"""
        
        result = await self._generate(prompt, provider)
        return result
    
    async def analyze_subreddit(
        self, 
        subreddit_name: str,
        posts: List[Dict[str, Any]],
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Analyze a subreddit based on its posts and provide insights.
        
        Args:
            subreddit_name: Name of the subreddit
            posts: List of posts data
            provider: Specific LLM provider to use (or None for default)
            
        Returns:
            Dict with subreddit analysis result
        """
        if not self.enabled or not self.clients:
            return {"analysis": "", "error": "LLM integration is disabled or no providers available"}
        
        # Create a concise representation of posts
        post_summaries = []
        for i, post in enumerate(posts[:10]):  # Use at most 10 posts
            post_summary = f"Post {i+1}: {post.get('title', '')} - Score: {post.get('score', 0)}, Comments: {post.get('num_comments', 0)}"
            post_summaries.append(post_summary)
        
        post_text = "\n".join(post_summaries)
        
        prompt = f"""Based on the following data from the r/{subreddit_name} subreddit, please provide:
1) A brief description of what this subreddit is about
2) The main topics/themes discussed in these posts
3) The overall tone/sentiment of the community

SUBREDDIT DATA:
{post_text}

ANALYSIS:"""
        
        result = await self._generate(prompt, provider)
        return result
    
    async def analyze_reddit_posts(
        self,
        posts: List[Dict[str, Any]],
        subreddit_name: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        max_posts: int = 500
    ) -> Dict[str, Any]:
        """
        Analyze a collection of Reddit posts and provide comprehensive insights.
        
        Args:
            posts: List of Reddit post data
            subreddit_name: Optional name of the subreddit
            provider: Specific LLM provider to use (or None for default)
            max_posts: Maximum number of posts to analyze
            
        Returns:
            Dict with analysis results including overview, topics, and insights
        """
        if not self.enabled or not self.clients:
            return {
                "overview": "", 
                "topics": [], 
                "insights": [],
                "error": "LLM integration is disabled or no providers available"
            }
        
        # Limit the number of posts
        posts = posts[:max_posts]
        
        # Prepare post content for analysis
        titles = [post.get('title', '') for post in posts if post.get('title')]
        titles_text = "\n".join([f"- {title}" for title in titles[:100]])
        
        # Create a sample of post content
        content_samples = []
        for i, post in enumerate(posts[:50]):
            if post.get('selftext'):
                # Truncate long posts
                selftext = post.get('selftext', '')
                if len(selftext) > 500:
                    selftext = selftext[:500] + "..."
                
                content_samples.append(f"Post {i+1}: {post.get('title', '')}\n{selftext}\n")
        
        content_text = "\n".join(content_samples)
        
        # Stats to provide context
        total_posts = len(posts)
        avg_score = sum(post.get('score', 0) for post in posts) / total_posts if total_posts else 0
        avg_comments = sum(post.get('num_comments', 0) for post in posts) / total_posts if total_posts else 0
        
        # Create the analysis prompt
        subreddit_info = f"r/{subreddit_name}" if subreddit_name else "these Reddit posts"
        
        prompt = f"""Analyze this collection of {total_posts} Reddit posts from {subreddit_info}.

POST STATISTICS:
- Total posts analyzed: {total_posts}
- Average score per post: {avg_score:.1f}
- Average comments per post: {avg_comments:.1f}

SAMPLE OF POST TITLES:
{titles_text}

SAMPLE OF POST CONTENT:
{content_text}

Based on these posts, please provide the following analysis in markdown format:

1) OVERVIEW: A concise overview (50-80 words) of what's happening in these posts - the main themes, patterns, and notable points.

2) TOPICS: A list of the 5-7 main topics of discussion, with a brief description of each topic. Format as a bulleted list using markdown.

3) INSIGHTS: 3-4 valuable insights expressed as one-line statements. Make each insight a single concise sentence that conveys a clear point. Format as a bulleted list using markdown.

Example format:
```
**OVERVIEW**
A concise 50-80 word overview of the content... 

**TOPICS**
- **Topic 1**: Description of the first major topic
- **Topic 2**: Description of the second major topic
- **Topic 3**: Description of the third major topic

**INSIGHTS**
- First key insight as a single sentence.
- Second key insight as a single sentence.
- Third key insight as a single sentence.
```

Use clear, direct language and focus on the most important patterns and trends.
"""
        
        # Generate the analysis
        result = await self._generate(prompt, provider, max_tokens=2000, temperature=0.3)
        
        # Process the result to extract the structured data
        if "text" in result and result["text"]:
            text = result["text"]
            
            # Extract sections using simple text parsing
            sections = {}
            current_section = None
            
            for line in text.split('\n'):
                line = line.strip()
                
                # Look for section headers
                if "OVERVIEW:" in line.upper() or line.upper() == "OVERVIEW":
                    current_section = "overview"
                    sections[current_section] = []
                elif "TOPICS:" in line.upper() or line.upper() == "TOPICS":
                    current_section = "topics" 
                    sections[current_section] = []
                elif "INSIGHTS:" in line.upper() or line.upper() == "INSIGHTS":
                    current_section = "insights"
                    sections[current_section] = []
                elif current_section and line:
                    # If we're in a section and the line has content, add it
                    sections[current_section].append(line)
            
            # Process each section
            overview = "\n".join(sections.get("overview", []))
            
            # Extract topics as a list
            topics = []
            for line in sections.get("topics", []):
                # Look for numbered or bulleted items
                if (line.startswith("-") or 
                    line.startswith("•") or 
                    line.startswith("*") or
                    (line[0].isdigit() and line[1:3] in (". ", ") "))):
                    topics.append(line.lstrip("- •*0123456789.) "))
            
            # Extract insights as a list
            insights = []
            for line in sections.get("insights", []):
                if (line.startswith("-") or 
                    line.startswith("•") or 
                    line.startswith("*") or
                    (line[0].isdigit() and line[1:3] in (". ", ") "))):
                    insights.append(line.lstrip("- •*0123456789.) "))
            
            return {
                "overview": overview,
                "topics": topics,
                "insights": insights,
                "provider": result.get("provider"),
                "model": result.get("model")
            }
        
        # If we couldn't process the text properly, return the raw text
        return {
            "overview": result.get("text", ""),
            "topics": [],
            "insights": [],
            "provider": result.get("provider"),
            "model": result.get("model"),
            "error": result.get("error")
        }
    
    async def batch_process(
        self,
        texts: List[str],
        operation: str = "summarize",
        provider: Optional[LLMProvider] = None,
        max_concurrent: int = 3,
        rate_limit_per_minute: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch with rate limiting.
        
        Args:
            texts: List of texts to process
            operation: Operation to perform ('summarize' or 'extract_insights')
            provider: Specific LLM provider to use (or None for default)
            max_concurrent: Maximum number of concurrent requests
            rate_limit_per_minute: Maximum requests per minute
            
        Returns:
            List of results for each text
        """
        if not self.enabled or not self.clients:
            error = {"error": "LLM integration is disabled or no providers available"}
            return [error] * len(texts)
        
        # Calculate sleep time between requests to respect rate limit
        sleep_time = 60 / rate_limit_per_minute
        
        # Create tasks for each text
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_rate_limit(text, index):
            async with semaphore:
                # Apply rate limiting
                if index > 0:
                    await asyncio.sleep(sleep_time)
                
                # Process based on operation
                if operation == "summarize":
                    return await self.summarize_text(text, provider)
                elif operation == "extract_insights":
                    return await self.extract_key_insights(text, provider)
                else:
                    return {"error": f"Unknown operation: {operation}"}
        
        # Create and gather tasks
        tasks = [
            process_with_rate_limit(text, i)
            for i, text in enumerate(texts)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def _generate(
        self, 
        prompt: str,
        provider: Optional[LLMProvider] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate text from a specific or default provider.
        
        Args:
            prompt: Input prompt
            provider: Specific provider to use (or None to use first available)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dict with generation results
        """
        if not self.enabled or not self.clients:
            return {"error": "LLM integration is disabled or no providers available", "text": ""}
        
        # Select provider
        client = None
        if provider and provider in self.clients:
            client = self.clients[provider]
        else:
            # Use first available provider
            for available_provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GOOGLE]:
                if available_provider in self.clients:
                    client = self.clients[available_provider]
                    break
        
        if not client:
            return {"error": "No LLM provider available", "text": ""}
        
        # Generate response
        return await client.generate(prompt, max_tokens, temperature)


# Create global instance
llm_service = LLMService() 