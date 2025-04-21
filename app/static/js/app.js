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
    
    // Dashboard Elements (Inline)
    const dashboardContainer = document.getElementById('analysis-dashboard-container');
    const dashboardLoading = document.getElementById('dashboard-loading');
    const dashboardContent = document.getElementById('dashboard-content');
    const dashboardSentimentChart = document.getElementById('dashboard-sentiment-chart');
    const dashboardKeywordsCloud = document.getElementById('dashboard-keywords-cloud');
    const dashboardPositiveCount = document.getElementById('dashboard-positive-count');
    const dashboardNegativeCount = document.getElementById('dashboard-negative-count');
    const dashboardNeutralCount = document.getElementById('dashboard-neutral-count');
    const dashboardTotalCount = document.getElementById('dashboard-total-count');
    const dashboardPositivePercentage = document.getElementById('dashboard-positive-percentage');
    const dashboardNegativePercentage = document.getElementById('dashboard-negative-percentage');
    const dashboardNeutralPercentage = document.getElementById('dashboard-neutral-percentage');
    
    // Analysis Modal Elements (for individual post analysis)
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
    let dashboardSentimentChartInstance = null;
    
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
            
            // Automatically analyze all content if results exist
            if (currentResults.length > 0) {
                analyzeAllContent();
            }
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
            dashboardContainer.style.display = 'none'; // Hide dashboard if no results
            return;
        }
        
        resultsTable.style.display = 'table';
        noResultsMessage.style.display = 'none';
        analyzeAllBtn.style.display = 'none'; // Hide the analyze all button since it's automatic
        
        // Create table rows
        results.forEach((post, index) => {
            const row = document.createElement('tr');
            
            // Format timestamp
            const createdDate = new Date(post.created_utc * 1000);
            const formattedDate = moment(createdDate).format('MMM D, YYYY h:mm A');
            
            // Determine if we have sentiment data
            let sentimentHtml = '';
            if (post.sentiment) {
                let sentimentClass = 'neutral';
                if (post.sentiment.label === 'positive') {
                    sentimentClass = 'positive';
                } else if (post.sentiment.label === 'negative') {
                    sentimentClass = 'negative';
                }
                sentimentHtml = `<span class="badge sentiment-${sentimentClass} ms-2">${post.sentiment.label}</span>`;
            }
            
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
                            ${sentimentHtml}
                        </div>
                    </div>
                </td>
                <td>r/${post.subreddit}</td>
                <td>u/${post.author}</td>
                <td title="${formattedDate}">${moment(createdDate).fromNow()}</td>
                <td>${post.score} <i class="bi bi-arrow-up-short text-muted"></i></td>
            `;
            
            resultsTbody.appendChild(row);
        });
        
        // No need to scroll to results here since dashboard will be shown above
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
                    method: "hybrid",
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
        // Check if we have results to analyze
        if (!currentResults || currentResults.length === 0) {
            showAlert('No posts to analyze', 'warning');
            return;
        }
        
        // Show loading state for the inline dashboard
        dashboardContainer.style.display = 'block';
        dashboardLoading.style.display = 'block';
        dashboardContent.style.display = 'none';
        
        // Scroll to dashboard since it's now above the results
        dashboardContainer.scrollIntoView({ behavior: 'smooth' });
        
        // Show loading state alert
        showAlert('Analyzing all posts, please wait...', 'info');
        
        // Extract content from all posts
        const textsToAnalyze = currentResults.map(post => {
            return post.selftext ? `${post.title}\n\n${post.selftext}` : post.title;
        });
        
        // Perform batch sentiment analysis
        fetch(`${apiBaseUrl}/analysis/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                operation: "sentiment",
                texts: textsToAnalyze
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Add sentiment data to posts
            currentResults.forEach((post, i) => {
                if (data.results && data.results[i]) {
                    post.sentiment = data.results[i];
                }
            });
            
            // Now perform batch keyword extraction
            return fetch(`${apiBaseUrl}/analysis/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    operation: "keywords",
                    texts: textsToAnalyze,
                    params: {
                        method: "hybrid",
                        num_keywords: 5
                    }
                })
            });
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Add keyword data to posts
            currentResults.forEach((post, i) => {
                if (data.results && data.results[i]) {
                    post.keywords = data.results[i];
                }
            });
            
            // Re-display results with sentiment info
            displayResults(currentResults);
            
            // Show dashboard with sentiment and keyword summary
            displayDashboard();
            
            // Show success message
            showAlert('All posts have been analyzed successfully!', 'success');
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert(`Failed to analyze all posts: ${error.message}`, 'danger');
            // Hide loading spinner on error
            dashboardLoading.style.display = 'none';
        });
    }
    
    /**
     * Display sentiment analysis results (for individual post modal)
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
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 8,
                            padding: 10
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Display extracted keywords (for individual post modal)
     * @param {Object} data - Keyword extraction results
     */
    function displayKeywords(data) {
        // Handle missing data
        if (!data || !data.keywords || data.keywords.length === 0) {
            console.log("No keywords found in data:", data);
            
            // Generate fallback keywords from content if needed
            const currentPost = document.getElementById('original-content').textContent;
            if (currentPost && currentPost.length > 0) {
                const fallbackKeywords = generateFallbackKeywords(currentPost);
                if (fallbackKeywords.length > 0) {
                    const keywordItems = fallbackKeywords.map((item, i) => {
                        // Calculate size based on frequency (higher index = less frequent)
                        const size = 0.75 - (i * 0.025); // Reduced by half from 1.5 and 0.05
                        return `
                            <div class="keyword-item" style="font-size: ${size}em;">
                                ${item.keyword} <small class="text-muted">(generated)</small>
                            </div>
                        `;
                    }).join('');
                    
                    keywordsCloud.innerHTML = keywordItems;
                    return;
                }
            }
            
            keywordsCloud.innerHTML = '<p class="text-muted">No significant keywords found. Try content with more text.</p>';
            return;
        }
        
        // Create keyword cloud
        const keywordItems = data.keywords.map((keyword, i) => {
            const score = data.scores && data.scores[i] ? data.scores[i] : 0.5;
            // Calculate size based on score (relative to highest score)
            const maxScore = Math.max(...(data.scores || [0.5]));
            const size = 0.5 + (score / maxScore) * 0.75; // Reduced by half from 1 and 1.5
            return `
                <div class="keyword-item" style="font-size: ${size}em;">
                    ${keyword}
                </div>
            `;
        }).join('');
        
        keywordsCloud.innerHTML = keywordItems;
    }
    
    /**
     * Display original content (for individual post modal)
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
    
    /**
     * Display dashboard with sentiment and keyword summary
     */
    function displayDashboard() {
        // Count sentiment categories
        let positive = 0, negative = 0, neutral = 0;
        let total = currentResults.length;
        
        // Collect all keywords for aggregation
        let keywordFrequency = {};
        
        currentResults.forEach(post => {
            // Count sentiment
            if (post.sentiment) {
                if (post.sentiment.label === 'positive') positive++;
                else if (post.sentiment.label === 'negative') negative++;
                else neutral++;
            }
            
            // Collect keywords
            if (post.keywords && post.keywords.keywords && post.keywords.keywords.length > 0) {
                post.keywords.keywords.forEach((keyword, i) => {
                    const score = post.keywords.scores[i] || 0.5;
                    if (!keywordFrequency[keyword]) {
                        keywordFrequency[keyword] = { count: 1, score: score };
                    } else {
                        keywordFrequency[keyword].count++;
                        keywordFrequency[keyword].score += score;
                    }
                });
            } else {
                // Generate fallback keywords if none were found
                const content = post.selftext ? `${post.title}\n\n${post.selftext}` : post.title;
                const fallbackKeywords = generateFallbackKeywords(content);
                
                fallbackKeywords.forEach(item => {
                    if (!keywordFrequency[item.keyword]) {
                        keywordFrequency[item.keyword] = { count: 1, score: 0.5 };
                    } else {
                        keywordFrequency[item.keyword].count++;
                    }
                });
            }
        });
        
        // Calculate percentages
        const posPercentage = Math.round((positive / total) * 100) || 0;
        const negPercentage = Math.round((negative / total) * 100) || 0;
        const neuPercentage = Math.max(0, 100 - posPercentage - negPercentage);
        
        // Update the dashboard sentiment summary
        dashboardPositiveCount.textContent = positive;
        dashboardNegativeCount.textContent = negative;
        dashboardNeutralCount.textContent = neutral;
        dashboardTotalCount.textContent = total;
        
        dashboardPositivePercentage.textContent = `${posPercentage}%`;
        dashboardNegativePercentage.textContent = `${negPercentage}%`;
        dashboardNeutralPercentage.textContent = `${neuPercentage}%`;
        
        // Create sentiment chart in the dashboard
        if (dashboardSentimentChartInstance) {
            dashboardSentimentChartInstance.destroy();
        }
        
        dashboardSentimentChartInstance = new Chart(dashboardSentimentChart, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [positive, negative, neutral],
                    backgroundColor: ['#43a047', '#e53935', '#757575'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            boxWidth: 8,
                            padding: 10
                        }
                    },
                    tooltip: {
                        displayColors: false
                    }
                }
            }
        });
        
        // Create merged keyword cloud
        let sortedKeywords = Object.keys(keywordFrequency).map(keyword => {
            return {
                keyword: keyword,
                count: keywordFrequency[keyword].count,
                score: keywordFrequency[keyword].score
            };
        });
        
        // Sort by count (frequency) and then by score
        sortedKeywords.sort((a, b) => {
            if (b.count !== a.count) return b.count - a.count;
            return b.score - a.score;
        });
        
        // Get top 20 keywords
        const topKeywords = sortedKeywords.slice(0, 20);
        
        // Update the dashboard keywords cloud
        if (topKeywords.length === 0) {
            dashboardKeywordsCloud.innerHTML = '<p class="text-muted">No significant keywords found across posts.</p>';
        } else {
            // Create keyword cloud
            const maxCount = Math.max(...topKeywords.map(k => k.count));
            const keywordItems = topKeywords.map(item => {
                // Calculate size based on frequency (relative to highest frequency)
                const size = 0.5 + (item.count / maxCount) * 0.75; // Reduced by half from 1 and 1.5
                const opacity = 0.6 + (item.count / maxCount) * 0.4;
                return `
                    <div class="keyword-item d-inline-block m-2" 
                         style="font-size: ${size}em; opacity: ${opacity};">
                        ${item.keyword} <small class="text-muted">(${item.count})</small>
                    </div>
                `;
            }).join('');
            
            dashboardKeywordsCloud.innerHTML = keywordItems;
        }
        
        // Show dashboard content, hide loading
        dashboardLoading.style.display = 'none';
        dashboardContent.style.display = 'block';
        
        // Scroll to dashboard when content is ready
        dashboardContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    /**
     * Generate fallback keywords from text content when API fails
     * @param {string} text - Text content
     * @returns {Array} Array of keyword objects
     */
    function generateFallbackKeywords(text) {
        // Basic stopwords list
        const stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                          'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                          'to', 'was', 'were', 'will', 'with', 'this', 'i', 'you', 'they',
                          'but', 'not', 'what', 'all', 'their', 'when', 'up', 'about',
                          'so', 'out', 'if', 'into', 'just', 'do', 'can', 'some'];
        
        // Tokenize and clean text
        const words = text.toLowerCase()
            .replace(/[^\w\s]/g, '') // Remove punctuation
            .split(/\s+/)            // Split on whitespace
            .filter(word => 
                word.length > 3 &&   // Only words longer than 3 chars
                !stopwords.includes(word) && // Remove stopwords
                !parseInt(word)      // Remove numbers
            );
        
        // Count word frequency
        const wordCounts = {};
        words.forEach(word => {
            wordCounts[word] = (wordCounts[word] || 0) + 1;
        });
        
        // Convert to array and sort by frequency
        const sortedWords = Object.keys(wordCounts)
            .map(keyword => ({ keyword, count: wordCounts[keyword] }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 10); // Get top 10 keywords
            
        return sortedWords;
    }
}); 