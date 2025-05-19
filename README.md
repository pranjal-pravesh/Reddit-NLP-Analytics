# Reddit Analysis

A scalable FastAPI backend for Reddit data retrieval and analysis with NLP capabilities and optional LLM integration.

## Features

- **Reddit Data Retrieval**
  - Authenticated access via PRAW or fallback to Reddit's public API
  - Search across all of Reddit or specific subreddits
  - Filter by time period and sort method
  - Support for pagination and result limits

- **NLP Analysis**
  - Sentiment analysis using CardiffNLP's Twitter-RoBERTa model
  - Keyword extraction via TF-IDF and TextRank
  - Topic modeling using LDA or BERTopic
  - Time-based post frequency histograms

- **LLM Integration (Optional)**
  - Support for OpenAI, Anthropic Claude, and Google Gemini
  - Text summarization and insights extraction
  - Keyword clustering
  - Provider-agnostic interface with batch processing

## Tech Stack

- FastAPI for async REST API
- Pydantic for data validation
- PRAW (Python Reddit API Wrapper)
- Hugging Face Transformers
- NLTK and scikit-learn for NLP
- Optional LLM integration (OpenAI, Anthropic, Google)

## Setup

### Prerequisites

- Python 3.9+
- Reddit API credentials (optional but recommended)
- LLM API keys (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reddit-analysis-api.git
cd reddit-analysis-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

5. Edit the `.env` file with your configuration:
```
# Application settings
APP_ENV=development
DEBUG=True
LOG_LEVEL=INFO

# Reddit API Credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=RedditAnalysisApp/1.0

# LLM settings (optional)
ENABLE_LLM_INTEGRATION=False
# Add API keys if needed
```

### Running the API

Start the server with:

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### Documentation

- Interactive API documentation: http://localhost:8000/docs
- ReDoc documentation: http://localhost:8000/redoc

### Reddit Endpoints

- `GET /api/v1/reddit/search`: Search Reddit posts
- `GET /api/v1/reddit/subreddit/{subreddit_name}`: Get subreddit info
- `GET /api/v1/reddit/post/{post_id}`: Get post details

### Analysis Endpoints

- `POST /api/v1/analysis/sentiment`: Analyze text sentiment
- `POST /api/v1/analysis/keywords`: Extract keywords from texts
- `POST /api/v1/analysis/topics`: Perform topic modeling
- `POST /api/v1/analysis/time-histogram`: Generate post frequency histogram
- `POST /api/v1/analysis/batch`: Process multiple texts

### LLM Endpoints (Optional)

- `POST /api/v1/analysis/llm/summarize`: Summarize text with an LLM
- `POST /api/v1/analysis/llm/cluster-keywords`: Cluster keywords into topics

## Usage Examples

### Search Reddit

```python
import requests

response = requests.get(
    "http://localhost:8000/api/v1/reddit/search",
    params={
        "query": "machine learning",
        "subreddit": "datascience",
        "sort": "relevance",
        "timeframe": "month",
        "page": 1,
        "page_size": 25
    }
)
results = response.json()
```

### Analyze Sentiment

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/analysis/sentiment",
    json={
        "text": "I really enjoyed this fascinating documentary about space exploration!"
    }
)
sentiment = response.json()
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [PRAW](https://praw.readthedocs.io/)
- [CardiffNLP](https://github.com/cardiffnlp/tweeteval)
- [Hugging Face Transformers](https://huggingface.co/transformers/) 
