# Reddit Analysis API

A full-stack web application for Reddit data retrieval and analysis with advanced NLP capabilities and optional LLM integration. Features a modern web interface built with FastAPI and interactive data visualization.

## 🚀 Features

- **Web Interface**
  - Interactive dashboard for Reddit data exploration
  - Real-time analysis results with visualizations
  - User-friendly forms for search configuration
  - Responsive design for desktop and mobile
  - Data export capabilities

- **Reddit Data Retrieval**
  - Authenticated access via PRAW or fallback to Reddit's public API
  - Search across all of Reddit or specific subreddits
  - Filter by time period and sort method
  - Support for pagination and result limits
  - Real-time data fetching with rate limiting

- **Advanced NLP Analysis**
  - Sentiment analysis using CardiffNLP's Twitter-RoBERTa model
  - Keyword extraction via TF-IDF and TextRank algorithms
  - Topic modeling using LDA or BERTopic
  - Time-based post frequency histograms
  - Batch processing for large datasets
  - Interactive charts and visualizations

- **LLM Integration (Optional)**
  - Support for OpenAI GPT, Anthropic Claude, and Google Gemini
  - Text summarization and insights extraction
  - Keyword clustering and semantic analysis
  - Provider-agnostic interface with batch processing

- **Production Features**
  - Full-stack FastAPI application with Jinja2 templates
  - RESTful API with comprehensive documentation
  - Comprehensive error handling and logging
  - Rate limiting and request validation
  - Configurable caching
  - Docker support

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript, Jinja2 Templates
- **Data Validation**: Pydantic v2
- **Reddit API**: PRAW (Python Reddit API Wrapper)
- **NLP**: Hugging Face Transformers, NLTK, scikit-learn, BERTopic
- **Visualization**: Chart.js, Plotly (web charts)
- **LLM Integration**: OpenAI, Anthropic, Google AI APIs
- **Database**: Optional PostgreSQL/SQLite support
- **Deployment**: Docker, Docker Compose

## 📋 Prerequisites

- Python 3.9+
- Reddit API credentials (optional but recommended for higher rate limits)
- LLM API keys (optional for advanced features)
- Modern web browser for the interface

## ⚡ Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd reddit-analysis-api

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section below)
# Windows
notepad .env
# macOS/Linux
nano .env
```

### 3. Run the Application

```bash
# Development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Access the application:**
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## ⚙️ Configuration

Edit your `.env` file with the following settings:

```env
# Application Settings
APP_ENV=development
DEBUG=True
LOG_LEVEL=INFO
API_V1_STR=/api/v1

# Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Reddit API Credentials (Optional but recommended)
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=RedditAnalysisApp/1.0 by YourUsername

# LLM Integration (Optional)
ENABLE_LLM_INTEGRATION=false

# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google Gemini
GOOGLE_API_KEY=your_google_key_here

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Caching
ENABLE_CACHE=true
CACHE_TTL=300
```

## 📁 Project Structure

```
reddit-analysis-platform/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── base.html          # Base template
│   │   ├── index.html         # Home page
│   │   ├── search.html        # Search interface
│   │   ├── analysis.html      # Analysis results
│   │   └── dashboard.html     # Analytics dashboard
│   ├── static/                # Static web assets
│   │   ├── css/              # Stylesheets
│   │   ├── js/               # JavaScript files
│   │   ├── images/           # Images and icons
│   │   └── charts/           # Chart configurations
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Logging setup
│   │   └── security.py        # Security utilities
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/     # API route handlers
│   │   │   └── __init__.py
│   │   └── dependencies.py    # API dependencies
│   ├── services/
│   │   ├── reddit_service.py  # Reddit API integration
│   │   ├── nlp_service.py     # NLP analysis services
│   │   └── llm_service.py     # LLM integration services
│   ├── utils/
│   │   ├── cache.py           # Caching utilities
│   │   ├── rate_limiter.py    # Rate limiting
│   │   └── visualization.py   # Chart generation
├── tests/                     # Test suite
├── docker/                    # Docker configuration
├── docs/                      # Additional documentation
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🌐 Web Interface

### Dashboard Features
- **Search Interface**: Easy-to-use forms for Reddit data retrieval
- **Real-time Analysis**: Live processing with progress indicators
- **Interactive Charts**: Sentiment trends, topic distributions, keyword clouds
- **Data Export**: Download results as CSV, JSON, or PDF reports
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Navigation
- `/` - Home page with quick start guide
- `/search` - Advanced Reddit search interface
- `/dashboard` - Analytics dashboard with visualizations
- `/analysis` - Detailed analysis results and reports
- `/api/v1/docs` - API documentation (Swagger UI)

## 📚 API Documentation

The platform provides both a web interface and a comprehensive REST API:

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Core Endpoints

#### Reddit Data Retrieval
```http
GET /api/v1/reddit/search
GET /api/v1/reddit/subreddit/{subreddit_name}
GET /api/v1/reddit/post/{post_id}
GET /api/v1/reddit/user/{username}
```

#### NLP Analysis
```http
POST /api/v1/analysis/sentiment
POST /api/v1/analysis/keywords
POST /api/v1/analysis/topics
POST /api/v1/analysis/time-histogram
POST /api/v1/analysis/batch
```

#### LLM Integration (Optional)
```http
POST /api/v1/analysis/llm/summarize
POST /api/v1/analysis/llm/cluster-keywords
POST /api/v1/analysis/llm/insights
```

## 💡 Usage Examples

### Web Interface Usage

1. **Navigate to the homepage** at http://localhost:8000
2. **Use the search form** to specify your Reddit search criteria
3. **View real-time results** as the analysis processes
4. **Explore interactive charts** showing sentiment trends and topics
5. **Export your data** in various formats for further analysis

### Basic Reddit Search (API)

```python
import requests

# Search for posts about artificial intelligence
response = requests.get(
    "http://localhost:8000/api/v1/reddit/search",
    params={
        "query": "artificial intelligence",
        "subreddit": "MachineLearning",
        "sort": "relevance",
        "timeframe": "week",
        "limit": 50
    }
)

posts = response.json()
print(f"Found {len(posts['data'])} posts")
```

### Web Interface Integration (JavaScript)

```javascript
// Example: Fetch and display sentiment analysis results
async function analyzeSentiment(texts) {
    const response = await fetch('/api/v1/analysis/sentiment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts: texts })
    });
    
    const results = await response.json();
    
    // Update the chart with new data
    updateSentimentChart(results);
}
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Docker

```bash
# Build image
docker build -t reddit-analysis-api .

# Run container
docker run -d \
  --name reddit-api \
  -p 8000:8000 \
  --env-file .env \
  reddit-analysis-api
```

## 🧪 Testing

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_reddit_service.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black app/ tests/
isort app/ tests/

# Run linting
flake8 app/ tests/
mypy app/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [PRAW](https://praw.readthedocs.io/)
- [CardiffNLP](https://github.com/cardiffnlp/tweeteval)
- [Hugging Face Transformers](https://huggingface.co/transformers/) 