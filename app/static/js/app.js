/**
 * Reddit Analysis Platform - Main JS
 * Handles Reddit data retrieval and content analysis
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let currentResults = [];
    const apiBaseUrl = '/api/v1';
    
    // DOM Elements
    const searchForm = document.getElementById('reddit-search-form');
    const searchBtn = document.getElementById('search-btn');
    const resultsContainer = document.getElementById('results-container');
    const resultsTable = document.getElementById('results-table');
    const resultsTbody = document.getElementById('results-tbody');
    const noResultsMessage = document.getElementById('no-results-message');
    const analyzeAllBtn = document.getElementById('analyze-all-btn');
    
    // Analysis Modal Elements
    const analysisModal = new bootstrap.Modal(document.getElementById('analysis-modal'));
    const analysisLoading = document.getElementById('analysis-loading');
    const analysisContent = document.getElementById('analysis-content');
    const sentimentChart = document.getElementById('sentiment-chart');
    const keywordsCloud = document.getElementById('keywords-cloud');
    const originalContent = document.getElementById('original-content');
    const sentimentScore = document.getElementById('sentiment-score');
    const sentimentExplanation = document.getElementById('sentiment-explanation');
    
    // Initialize Chart.js
    let sentimentChartInstance = null;
    
    // Event Listeners
    if (searchForm) {
        searchForm.addEventListener('submit', handleSearch);
    }
    if (analyzeAllBtn) {
        analyzeAllBtn.addEventListener('click', analyzeAllContent);
    }
    
    /**
     * Handle search form submission
     * @param {Event} e - Form submission event
     */
    async function handleSearch(e) {
        e.preventDefault();
        
        // Update button state
        searchBtn.disabled = true;
        searchBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
        
        // Get form values
        const searchType = document.getElementById('search-type').value;
        const searchQuery = document.getElementById('search-query').value.trim();
        const timeFilter = document.getElementById('time-filter').value;
        const sortBy = document.getElementById('sort-by').value;
        const limit = document.getElementById('limit').value;
        
        // Validate form
        if (!searchQuery) {
            showAlert('Please enter a search query', 'danger');
            resetSearchButton();
            return;
        }
        
        try {
            // Build request based on search type
            let endpoint = '';
            let requestData = {};
            
            switch (searchType) {
                case 'subreddit':
                    endpoint = `${apiBaseUrl}/reddit/subreddit`;
                    requestData = {
                        subreddit: searchQuery,
                        sort: sortBy,
                        time_filter: timeFilter,
                        limit: parseInt(limit)
                    };
                    break;
                    
                case 'user':
                    endpoint = `${apiBaseUrl}/reddit/user`;
                    requestData = {
                        username: searchQuery,
                        sort: sortBy,
                        time_filter: timeFilter,
                        limit: parseInt(limit)
                    };
                    break;
                    
                case 'search':
                    endpoint = `${apiBaseUrl}/reddit/search`;
                    requestData = {
                        query: searchQuery,
                        sort: sortBy,
                        time_filter: timeFilter,
                        limit: parseInt(limit)
                    };
                    break;
            }
            
            // Make API request
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            currentResults = data.posts || [];
            
            // Display results
            displayResults(currentResults);
        } catch (error) {
            console.error('Error:', error);
            showAlert(`Failed to fetch data: ${error.message}`, 'danger');
        } finally {
            resetSearchButton();
        }
    }
    
    /**
     * Display search results in the table
     * @param {Array} results - Array of Reddit post objects
     */
    function displayResults(results) {
        resultsContainer.style.display = 'block';
        resultsTbody.innerHTML = '';
        
        if (results.length === 0) {
            resultsTable.style.display = 'none';
            noResultsMessage.style.display = 'block';
            analyzeAllBtn.style.display = 'none';
            return;
        }
        
        resultsTable.style.display = 'table';
        noResultsMessage.style.display = 'none';
        analyzeAllBtn.style.display = 'inline-block';
        
        // Create table rows
        results.forEach((post, index) => {
            const row = document.createElement('tr');
            
            // Format timestamp
            const createdDate = new Date(post.created_utc * 1000);
            const formattedDate = moment(createdDate).format('MMM D, YYYY h:mm A');
            
            row.innerHTML = `
                <td>
                    <div class="d-flex align-items-center">
                        ${post.thumbnail && post.thumbnail !== 'self' && post.thumbnail !== 'default' ? 
                            `<img src="${post.thumbnail}" class="me-2" style="width: 60px; height: 40px; object-fit: cover; border-radius: 4px;" alt="Thumbnail">` : 
                            `<div class="me-2 text-center" style="width: 60px; height: 40px; background: #eee; border-radius: 4px; display: flex; align-items: center; justify-content: center;">
                                <i class="bi bi-card-text text-muted"></i>
                            </div>`
                        }
                        <div>
                            <a href="${post.url}" target="_blank" class="fw-bold text-decoration-none">
                                ${truncateText(post.title, 80)}
                            </a>
                            ${post.is_self ? '<span class="badge bg-secondary ms-1">Self Post</span>' : ''}
                        </div>
                    </div>
                </td>
                <td>r/${post.subreddit}</td>
                <td>u/${post.author}</td>
                <td title="${formattedDate}">${moment(createdDate).fromNow()}</td>
                <td>${post.score} <i class="bi bi-arrow-up-short text-muted"></i></td>
                <td>
                    <button class="btn btn-sm btn-outline-primary analyze-btn" data-index="${index}">
                        <i class="bi bi-graph-up"></i> Analyze
                    </button>
                </td>
            `;
            
            resultsTbody.appendChild(row);
        });
        
        // Add event listeners to analyze buttons
        document.querySelectorAll('.analyze-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const index = parseInt(btn.dataset.index);
                analyzeContent(currentResults[index]);
            });
        });
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    /**
     * Analyze a single post's content
     * @param {Object} post - Reddit post object
     */
    async function analyzeContent(post) {
        // Show modal and loading state
        analysisModal.show();
        analysisLoading.style.display = 'block';
        analysisContent.style.display = 'none';
        
        try {
            // Get content to analyze (title + selftext or title only)
            const content = post.selftext ? `${post.title}\n\n${post.selftext}` : post.title;
            
            // Perform sentiment analysis
            const sentimentResponse = await fetch(`${apiBaseUrl}/analysis/sentiment`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: content })
            });
            
            if (!sentimentResponse.ok) {
                throw new Error(`Sentiment API error: ${sentimentResponse.status}`);
            }
            
            const sentimentData = await sentimentResponse.json();
            
            // Perform keyword extraction
            const keywordResponse = await fetch(`${apiBaseUrl}/analysis/keywords`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    texts: [content],
                    method: "tfidf",
                    num_keywords: 10
                })
            });
            
            if (!keywordResponse.ok) {
                throw new Error(`Keyword API error: ${keywordResponse.status}`);
            }
            
            const keywordData = await keywordResponse.json();
            
            // Display analysis results
            displaySentimentAnalysis(sentimentData);
            displayKeywords(keywordData.results[0]);
            displayOriginalContent(post);
            
            // Hide loading, show content
            analysisLoading.style.display = 'none';
            analysisContent.style.display = 'block';
        } catch (error) {
            console.error('Analysis error:', error);
            analysisLoading.style.display = 'none';
            analysisContent.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    Failed to analyze content: ${error.message}
                </div>
            `;
            analysisContent.style.display = 'block';
        }
    }
    
    /**
     * Analyze all posts in bulk
     */
    function analyzeAllContent() {
        // Implement bulk analysis if needed
        showAlert('Bulk analysis feature is coming soon!', 'info');
    }
    
    /**
     * Display sentiment analysis results
     * @param {Object} data - Sentiment analysis data
     */
    function displaySentimentAnalysis(data) {
        // Update sentiment score display
        const scorePercentage = Math.round(data.score * 100);
        let sentimentClass = 'neutral';
        let sentimentColor = '#757575';
        
        if (data.label === 'positive') {
            sentimentClass = 'positive';
            sentimentColor = '#43a047';
        } else if (data.label === 'negative') {
            sentimentClass = 'negative';
            sentimentColor = '#e53935';
        }
        
        sentimentScore.innerHTML = `
            <div class="text-center mb-3">
                <div class="display-4 sentiment-${sentimentClass}">
                    ${scorePercentage}%
                </div>
                <div class="mt-2 text-capitalize fw-bold sentiment-${sentimentClass}">
                    ${data.label}
                </div>
            </div>
        `;
        
        // Add explanation
        sentimentExplanation.innerHTML = `
            <p>This content has been analyzed as <strong class="sentiment-${sentimentClass}">${data.label}</strong> 
            with ${scorePercentage}% confidence.</p>
            <p class="mb-0 small text-muted">Analysis performed using CardiffNLP's Twitter-RoBERTa model.</p>
        `;
        
        // Create or update chart
        if (sentimentChartInstance) {
            sentimentChartInstance.destroy();
        }
        
        sentimentChartInstance = new Chart(sentimentChart, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [
                        data.positive || 0,
                        data.negative || 0,
                        data.neutral || 0
                    ],
                    backgroundColor: ['#43a047', '#e53935', '#757575'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    /**
     * Display extracted keywords
     * @param {Object} data - Keyword extraction results
     */
    function displayKeywords(data) {
        if (!data || !data.keywords || data.keywords.length === 0) {
            keywordsCloud.innerHTML = '<p class="text-muted">No significant keywords found.</p>';
            return;
        }
        
        // Create keyword cloud
        const keywordItems = data.keywords.map((keyword, i) => {
            const score = data.scores[i];
            // Calculate size based on score (relative to highest score)
            const size = 1 + (score / Math.max(...data.scores)) * 1.5;
            return `
                <div class="keyword-item" style="font-size: ${size}em;">
                    ${keyword}
                </div>
            `;
        }).join('');
        
        keywordsCloud.innerHTML = keywordItems;
    }
    
    /**
     * Display original content
     * @param {Object} post - Reddit post object
     */
    function displayOriginalContent(post) {
        // Create post display
        const content = post.selftext ? simpleMarkdown(post.selftext) : '<p class="text-muted">(No text content)</p>';
        
        originalContent.innerHTML = `
            <h5 class="mb-3">${post.title}</h5>
            <div class="post-metadata mb-3 small text-muted">
                <span><i class="bi bi-person me-1"></i> u/${post.author}</span>
                <span class="ms-3"><i class="bi bi-chat me-1"></i> ${post.num_comments} comments</span>
                <span class="ms-3"><i class="bi bi-arrow-up me-1"></i> ${post.score} points</span>
                <span class="ms-3"><i class="bi bi-clock me-1"></i> ${moment(new Date(post.created_utc * 1000)).fromNow()}</span>
            </div>
            <div class="post-content">
                ${content}
            </div>
            ${post.url && !post.is_self ? `
                <div class="mt-3">
                    <a href="${post.url}" target="_blank" class="btn btn-sm btn-outline-primary">
                        <i class="bi bi-link-45deg me-1"></i> Open Link
                    </a>
                </div>
            ` : ''}
        `;
    }
    
    /**
     * Reset search button to initial state
     */
    function resetSearchButton() {
        searchBtn.disabled = false;
        searchBtn.innerHTML = '<i class="bi bi-search me-2"></i>Fetch Data';
    }
    
    /**
     * Show alert message
     * @param {string} message - Alert message
     * @param {string} type - Alert type (success, danger, warning, info)
     */
    function showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Insert before search form
        if (searchForm) {
            searchForm.parentNode.insertBefore(alertDiv, searchForm);
        } else {
            document.querySelector('.container').prepend(alertDiv);
        }
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 300);
        }, 5000);
    }
    
    /**
     * Truncate text to specified length
     * @param {string} text - Text to truncate
     * @param {number} maxLength - Maximum length
     * @returns {string} Truncated text
     */
    function truncateText(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
    
    /**
     * Simple markdown converter
     * @param {string} text - Markdown text
     * @returns {string} HTML
     */
    function simpleMarkdown(text) {
        if (!text) return '';
        return text
            .replace(/\n\n/g, '<br><br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>')
            .replace(/^> (.*?)$/gm, '<blockquote>$1</blockquote>');
    }
}); 