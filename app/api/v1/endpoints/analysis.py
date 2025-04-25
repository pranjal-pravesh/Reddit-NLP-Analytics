from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.api.v1.schemas.analysis import (
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    KeywordExtractionRequest,
    KeywordExtractionResponse,
    LLMKeywordClusterRequest,
    LLMKeywordClusterResponse,
    LLMSummaryRequest,
    LLMSummaryResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    TimeHistogramRequest,
    TimeHistogramResponse,
    TopicModelingRequest,
    TopicModelingResponse,
)
from app.core.config import settings
from app.services.llm_service import LLMProvider, llm_service
from app.services.nlp_service import nlp_service
from app.utils.logging_config import logger
import psutil
import sys
import os
from datetime import datetime

router = APIRouter()


@router.post("/sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest = Body(...)
):
    """
    Analyze the sentiment of a text using CardiffNLP's Twitter-RoBERTa model.
    Returns sentiment label and confidence score.
    """
    try:
        result = nlp_service.analyze_sentiment(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}"
        )


@router.post("/sentiment-batch", response_model=List[Dict[str, Any]])
async def analyze_sentiment_batch(
    request: dict = Body(...)
):
    """
    Analyze the sentiment of multiple texts in a single request.
    Much more efficient than multiple individual calls.
    Returns list of sentiment analysis results.
    """
    try:
        if "texts" not in request or not isinstance(request["texts"], list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must include a 'texts' field containing a list of strings"
            )
            
        results = nlp_service.analyze_sentiment_batch(request["texts"])
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiments in batch: {str(e)}"
        )


@router.post("/keywords", response_model=KeywordExtractionResponse)
async def extract_keywords(
    request: KeywordExtractionRequest = Body(...)
):
    """
    Extract keywords from a list of texts using either TF-IDF or TextRank.
    Returns keywords and their scores for each input text.
    """
    try:
        # Validate input data
        if not request.texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No texts provided for keyword extraction"
            )
            
        # Convert any non-string items to strings and flatten if needed
        cleaned_texts = []
        for text in request.texts:
            if isinstance(text, list):
                # If text is a list, join elements with spaces
                cleaned_texts.append(" ".join(str(item) for item in text))
            else:
                # Otherwise, convert to string
                cleaned_texts.append(str(text))
        
        # Call NLP service with cleaned inputs
        results = nlp_service.extract_keywords(
            texts=cleaned_texts,
            method=request.method,
            num_keywords=request.num_keywords
        )
        
        # Validate results
        if not results:
            logger.warning("NLP service returned empty results for keyword extraction")
            return {"results": [{"keywords": [], "scores": []}] * len(cleaned_texts)}
            
        return {"results": results}
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting keywords: {str(e)}"
        )


@router.post("/topics", response_model=TopicModelingResponse)
async def model_topics(
    request: TopicModelingRequest = Body(...)
):
    """
    Perform topic modeling on a collection of texts using LDA or BERTopic.
    Returns identified topics and their keywords.
    """
    try:
        result = nlp_service.topic_modeling(
            texts=request.texts,
            num_topics=request.num_topics,
            method=request.method
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in topic modeling: {str(e)}"
        )


@router.post("/time-histogram", response_model=TimeHistogramResponse)
async def generate_time_histogram(
    request: TimeHistogramRequest = Body(...)
):
    """
    Generate a time-based histogram of post frequency.
    Returns counts, labels, and a base64-encoded plot image.
    """
    try:
        result = nlp_service.generate_time_histogram(
            timestamps=request.timestamps,
            interval=request.interval
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating time histogram: {str(e)}"
        )


@router.get("/optimization-status", response_model=Dict[str, Any])
async def check_optimization_status():
    """
    Check the status of various optimizations.
    Returns details about which optimizations are enabled and active.
    """
    try:
        status = nlp_service.get_optimization_status()
        
        # Add memory usage info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        status["memory_usage_mb"] = memory_info.rss / (1024 * 1024)
        status["virtual_memory_mb"] = memory_info.vms / (1024 * 1024)
        
        # Add timing information from cached properties
        status["system_info"] = {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
        }
        
        return {
            "status": "ok",
            "optimizations": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking optimization status: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/batch", response_model=BatchAnalysisResponse)
async def batch_analyze(
    request: BatchAnalysisRequest = Body(...)
):
    """
    Process multiple texts with the specified analysis operation.
    Supports sentiment analysis, keyword extraction, and topic modeling.
    """
    import time
    start_time = time.time()
    logger.info(f"Batch analysis request for operation: {request.operation}, {len(request.texts)} texts")
    
    try:
        if request.operation == "sentiment":
            # Use the optimized batch sentiment analysis
            logger.info("Using optimized batch sentiment analysis")
            results = nlp_service.analyze_sentiment_batch(request.texts)
            
        elif request.operation == "keywords":
            method = request.params.get("method", "tfidf") if request.params else "tfidf"
            num_keywords = request.params.get("num_keywords", 10) if request.params else 10
            
            # For large collections, force the hashing vectorizer
            if len(request.texts) > 500 and method == "tfidf":
                logger.info(f"Large collection detected ({len(request.texts)} texts): Using tfidf_hash method for better performance")
                method = "tfidf_hash"
                
            logger.info(f"Extracting keywords with method: {method}, num_keywords: {num_keywords}")
            results = nlp_service.extract_keywords(
                texts=request.texts,
                method=method,
                num_keywords=num_keywords
            )
            
        elif request.operation == "topics":
            if len(request.texts) < 5:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Topic modeling requires at least 5 texts"
                )
                
            method = request.params.get("method", "lda") if request.params else "lda"
            num_topics = request.params.get("num_topics", 5) if request.params else 5
            
            logger.info(f"Performing topic modeling with method: {method}, num_topics: {num_topics}")
            result = nlp_service.topic_modeling(
                texts=request.texts,
                method=method,
                num_topics=num_topics
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Batch {request.operation} completed in {elapsed_time:.2f} seconds")
            return {"results": [result], "operation": request.operation}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported operation: {request.operation}"
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch {request.operation} completed in {elapsed_time:.2f} seconds for {len(request.texts)} texts ({len(request.texts)/elapsed_time:.2f} texts/sec)")
        return {"results": results, "operation": request.operation}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch analysis: {str(e)}"
        )


# LLM Integration Endpoints (Optional)
@router.post("/llm/summarize", response_model=LLMSummaryResponse)
async def summarize_text(
    request: LLMSummaryRequest = Body(...)
):
    """
    Generate a concise summary of text using an LLM.
    Requires LLM integration to be enabled in settings.
    """
    if not settings.ENABLE_LLM_INTEGRATION:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM integration is disabled"
        )
        
    try:
        # Convert string provider to enum if specified
        provider = None
        if request.provider:
            try:
                provider = LLMProvider(request.provider)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid provider: {request.provider}"
                )
        
        result = await llm_service.summarize_text(
            text=request.text,
            provider=provider,
            max_length=request.max_length
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error summarizing text: {str(e)}"
        )


@router.post("/llm/cluster-keywords", response_model=LLMKeywordClusterResponse)
async def cluster_keywords(
    request: LLMKeywordClusterRequest = Body(...)
):
    """
    Cluster keywords into meaningful groups using an LLM.
    Requires LLM integration to be enabled in settings.
    """
    if not settings.ENABLE_LLM_INTEGRATION:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM integration is disabled"
        )
        
    try:
        # Convert string provider to enum if specified
        provider = None
        if request.provider:
            try:
                provider = LLMProvider(request.provider)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid provider: {request.provider}"
                )
        
        result = await llm_service.cluster_keywords(
            keywords=request.keywords,
            provider=provider
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clustering keywords: {str(e)}"
        ) 