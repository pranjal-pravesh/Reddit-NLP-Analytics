import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import analysis, reddit
from app.core.config import settings
from app.services.llm_service import llm_service
from app.services.reddit_client import reddit_client
from app.utils.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting up Reddit Analysis API...")
    
    # Yield control to the application
    yield
    
    # Shutdown
    logger.info("Shutting down Reddit Analysis API...")
    await reddit_client.close()
    await llm_service.close()


# Create FastAPI application
app = FastAPI(
    title="Reddit Analysis API",
    description="API for Reddit data retrieval and analysis with NLP and optional LLM integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler for custom error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": settings.APP_ENV
    }


# Include API routers
app.include_router(
    reddit.router,
    prefix="/api/v1/reddit",
    tags=["Reddit"]
)

app.include_router(
    analysis.router,
    prefix="/api/v1/analysis",
    tags=["Analysis"]
)


if __name__ == "__main__":
    """Run the application with uvicorn when script is executed directly"""
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    ) 