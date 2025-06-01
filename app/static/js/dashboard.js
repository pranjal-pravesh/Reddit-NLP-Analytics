/**
 * Reddit Analysis Platform - Dashboard JS
 * Handles data visualization and dashboard functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    const apiBaseUrl = '/api/v1';
    let currentData = null;
    
    // DOM Elements
    const dashboardDataset = document.getElementById('dashboard-dataset');
    const refreshDashboardBtn = document.getElementById('refresh-dashboard');
    const totalPosts = document.getElementById('total-posts');
    const totalComments = document.getElementById('total-comments');
    const uniqueAuthors = document.getElementById('unique-authors');
    const totalScore = document.getElementById('total-score');
    const topPostsFilter = document.getElementById('top-posts-filter');
    const topPostsTbody = document.getElementById('top-posts-tbody');
    
    // LLM Analysis Elements
    const runLlmAnalysisBtn = document.getElementById('run-llm-analysis');
    const llmProviderSelect = document.getElementById('llm-provider');
    const llmLoading = document.getElementById('llm-loading');
    const llmContent = document.getElementById('llm-content');
    const llmPlaceholder = document.getElementById('llm-placeholder');
    const llmError = document.getElementById('llm-error');
    const llmErrorMessage = document.getElementById('llm-error-message');
    const llmOverview = document.getElementById('llm-overview');
    const llmTopics = document.getElementById('llm-topics');
    const llmInsights = document.getElementById('llm-insights');
    const llmModelInfo = document.getElementById('llm-model-info');
    
    // Chart Elements
    const timeChart = document.getElementById('time-chart');
    const sentimentPieChart = document.getElementById('sentiment-pie-chart');
    const keywordsChart = document.getElementById('keywords-chart');
    const topicsChart = document.getElementById('topics-chart');
    const subredditChart = document.getElementById('subreddit-chart');
    
    // Chart Instances
    let timeChartInstance = null;
    let sentimentPieChartInstance = null;
    let keywordsChartInstance = null;
    let topicsChartInstance = null;
    let subredditChartInstance = null;
    
    // Event Listeners
    if (dashboardDataset) {
        dashboardDataset.addEventListener('change', loadDataset);
    }
    if (refreshDashboardBtn) {
        refreshDashboardBtn.addEventListener('click', refreshDashboard);
    }
    if (topPostsFilter) {
        topPostsFilter.addEventListener('change', filterTopPosts);
    }
    if (runLlmAnalysisBtn) {
        runLlmAnalysisBtn.addEventListener('click', generateLlmAnalysis);
    }
    
    // Initialize with mock data for demo
    initializeMockData();
    
    /**
     * Load selected dataset
     */
    function loadDataset() {
        const datasetId = dashboardDataset.value;
        
        if (!datasetId) return;
        
        // Show loading
        showLoading();
        
        // In a real implementation, this would fetch from server
        // For demo, just use mock data
        setTimeout(() => {
            // Process data
            updateDashboard(currentData);
            
            // Hide loading
            hideLoading();
        }, 1000);
    }
    
    /**
     * Refresh dashboard data
     */
    function refreshDashboard() {
        showLoading();
        
        // In a real implementation, this would refresh from server
        // For demo, just re-render with existing data
        setTimeout(() => {
            if (currentData) {
                updateDashboard(currentData);
            }
            hideLoading();
        }, 1000);
    }
    
    /**
     * Filter top posts based on selected criteria
     */
    function filterTopPosts() {
        const filterType = topPostsFilter.value;
        
        if (!currentData || !currentData.posts) return;
        
        let sortedPosts = [...currentData.posts];
        
        // Sort based on filter
        switch (filterType) {
            case 'score':
                sortedPosts.sort((a, b) => b.score - a.score);
                break;
            case 'comments':
                sortedPosts.sort((a, b) => b.num_comments - a.num_comments);
                break;
            case 'recency':
                sortedPosts.sort((a, b) => b.created_utc - a.created_utc);
                break;
        }
        
        // Display top 10 posts
        displayTopPosts(sortedPosts.slice(0, 10));
    }
    
    /**
     * Display top posts in table
     * @param {Array} posts - Array of post objects
     */
    function displayTopPosts(posts) {
        if (!topPostsTbody) return;
        
        topPostsTbody.innerHTML = '';
        
        posts.forEach(post => {
            const row = document.createElement('tr');
            
            // Format timestamp
            const createdDate = new Date(post.created_utc * 1000);
            const formattedDate = moment(createdDate).format('MMM D, YYYY');
            
            // Get sentiment class
            let sentimentClass = 'neutral';
            let sentimentLabel = 'Neutral';
            
            if (post.sentiment) {
                sentimentLabel = post.sentiment.label;
                if (post.sentiment.label === 'positive') {
                    sentimentClass = 'positive';
                } else if (post.sentiment.label === 'negative') {
                    sentimentClass = 'negative';
                }
            }
            
            row.innerHTML = `
                <td>
                    <div class="d-flex align-items-center">
                        <div>
                            <a href="${post.url}" target="_blank" class="fw-bold text-decoration-none">
                                ${truncateText(post.title, 70)}
                            </a>
                        </div>
                    </div>
                </td>
                <td>r/${post.subreddit}</td>
                <td>${post.score}</td>
                <td>${post.num_comments}</td>
                <td>${formattedDate}</td>
                <td class="sentiment-${sentimentClass}">${sentimentLabel}</td>
            `;
            
            topPostsTbody.appendChild(row);
        });
    }
    
    /**
     * Update dashboard with data
     * @param {Object} data - Dashboard data
     */
    function updateDashboard(data) {
        if (!data) return;
        
        // Update counter cards
        updateCounters(data);
        
        // Update charts
        updateTimeChart(data.timeData);
        updateSentimentChart(data.sentimentData);
        updateKeywordsChart(data.keywordData);
        updateTopicsChart(data.topicData);
        updateSubredditChart(data.subredditData);
        
        // Update top posts table
        filterTopPosts();
    }
    
    /**
     * Update dashboard counter cards
     * @param {Object} data - Dashboard data
     */
    function updateCounters(data) {
        if (totalPosts) totalPosts.textContent = data.totalPosts.toLocaleString();
        if (totalComments) totalComments.textContent = data.totalComments.toLocaleString();
        if (uniqueAuthors) uniqueAuthors.textContent = data.uniqueAuthors.toLocaleString();
        if (totalScore) totalScore.textContent = data.totalScore.toLocaleString();
    }
    
    /**
     * Update time chart
     * @param {Object} data - Time chart data
     */
    function updateTimeChart(data) {
        if (!timeChart) return;
        
        // Destroy existing chart
        if (timeChartInstance) {
            timeChartInstance.destroy();
        }
        
        // Create new chart
        timeChartInstance = new Chart(timeChart, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Posts',
                    data: data.values,
                    backgroundColor: 'rgba(30, 136, 229, 0.1)',
                    borderColor: 'rgba(30, 136, 229, 1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    pointBackgroundColor: 'rgba(30, 136, 229, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                }
            }
        });
    }
    
    /**
     * Update sentiment pie chart
     * @param {Object} data - Sentiment data
     */
    function updateSentimentChart(data) {
        if (!sentimentPieChart) return;
        
        // Destroy existing chart
        if (sentimentPieChartInstance) {
            sentimentPieChartInstance.destroy();
        }
        
        // Create new chart
        sentimentPieChartInstance = new Chart(sentimentPieChart, {
            type: 'pie',
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: data.colors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const total = context.dataset.data.reduce((sum, val) => sum + val, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${context.label}: ${percentage}% (${value})`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update keywords chart
     * @param {Object} data - Keyword data
     */
    function updateKeywordsChart(data) {
        if (!keywordsChart) return;
        
        // Destroy existing chart
        if (keywordsChartInstance) {
            keywordsChartInstance.destroy();
        }
        
        // Create new chart
        keywordsChartInstance = new Chart(keywordsChart, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Frequency',
                    data: data.values,
                    backgroundColor: 'rgba(0, 172, 193, 0.7)',
                    borderColor: 'rgba(0, 172, 193, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    },
                    x: {
                        ticks: {
                            autoSkip: false,
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    /**
     * Update topics chart
     * @param {Object} data - Topic data
     */
    function updateTopicsChart(data) {
        if (!topicsChart) return;
        
        // Destroy existing chart
        if (topicsChartInstance) {
            topicsChartInstance.destroy();
        }
        
        // Create new chart
        topicsChartInstance = new Chart(topicsChart, {
            type: 'radar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Topic Relevance',
                    data: data.values,
                    backgroundColor: 'rgba(255, 160, 0, 0.2)',
                    borderColor: 'rgba(255, 160, 0, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(255, 160, 0, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 1,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        angleLines: {
                            display: true
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update subreddit chart
     * @param {Object} data - Subreddit data
     */
    function updateSubredditChart(data) {
        if (!subredditChart) return;
        
        // Destroy existing chart
        if (subredditChartInstance) {
            subredditChartInstance.destroy();
        }
        
        // Create new chart
        subredditChartInstance = new Chart(subredditChart, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Posts',
                    data: data.values,
                    backgroundColor: 'rgba(126, 87, 194, 0.7)',
                    borderColor: 'rgba(126, 87, 194, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                indexAxis: 'y'
            }
        });
    }
    
    /**
     * Generate LLM analysis for the current data
     */
    async function generateLlmAnalysis() {
        if (!currentData || !currentData.posts || currentData.posts.length === 0) {
            showLlmError("No data available for analysis. Please load data first.");
            return;
        }
        
        // Show loading state
        showLlmLoading();
        
        try {
            // Prepare data for analysis
            const provider = llmProviderSelect.value || null;
            const subredditName = currentData.subredditName || null;
            
            // Prepare request data
            const requestData = {
                posts: currentData.posts,
                provider: provider,
                subreddit_name: subredditName,
                max_posts: Math.min(currentData.posts.length, 500) // Limit to 500 posts
            };
            
            // Make API request
            const response = await fetch(`${apiBaseUrl}/analysis/llm/analyze-reddit-posts`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to generate LLM analysis');
            }
            
            // Process response
            const data = await response.json();
            
            // Display results
            displayLlmResults(data);
            
        } catch (error) {
            console.error('Error generating LLM analysis:', error);
            showLlmError(error.message || 'Failed to generate LLM analysis');
        }
    }
    
    /**
     * Display LLM analysis results
     * @param {Object} data - LLM analysis data
     */
    function displayLlmResults(data) {
        // Set overview
        llmOverview.textContent = data.overview || '';
        
        // Set model info
        let modelInfo = 'Default LLM';
        if (data.provider && data.model) {
            modelInfo = `${data.provider} (${data.model})`;
        }
        llmModelInfo.textContent = modelInfo;
        
        // Clear previous topics and insights
        llmTopics.innerHTML = '';
        llmInsights.innerHTML = '';
        
        // Add topics
        if (data.topics && data.topics.length > 0) {
            data.topics.forEach(topic => {
                const topicElement = document.createElement('div');
                topicElement.className = 'list-group-item';
                topicElement.textContent = topic;
                llmTopics.appendChild(topicElement);
            });
        } else {
            const noTopics = document.createElement('div');
            noTopics.className = 'list-group-item text-muted';
            noTopics.textContent = 'No topics identified';
            llmTopics.appendChild(noTopics);
        }
        
        // Add insights
        if (data.insights && data.insights.length > 0) {
            data.insights.forEach(insight => {
                const insightElement = document.createElement('div');
                insightElement.className = 'list-group-item';
                insightElement.textContent = insight;
                llmInsights.appendChild(insightElement);
            });
        } else {
            const noInsights = document.createElement('div');
            noInsights.className = 'list-group-item text-muted';
            noInsights.textContent = 'No insights provided';
            llmInsights.appendChild(noInsights);
        }
        
        // Show content
        showLlmContent();
    }
    
    /**
     * Show LLM loading state
     */
    function showLlmLoading() {
        llmContent.classList.add('d-none');
        llmPlaceholder.classList.add('d-none');
        llmError.classList.add('d-none');
        llmLoading.classList.remove('d-none');
    }
    
    /**
     * Show LLM content
     */
    function showLlmContent() {
        llmLoading.classList.add('d-none');
        llmPlaceholder.classList.add('d-none');
        llmError.classList.add('d-none');
        llmContent.classList.remove('d-none');
    }
    
    /**
     * Show LLM error message
     * @param {String} message - Error message
     */
    function showLlmError(message) {
        llmLoading.classList.add('d-none');
        llmContent.classList.add('d-none');
        llmPlaceholder.classList.add('d-none');
        
        llmErrorMessage.textContent = message;
        llmError.classList.remove('d-none');
    }
    
    /**
     * Initialize mock data for demonstration
     */
    function initializeMockData() {
        // Create mock data for visualization
        const mockPosts = generateMockPosts(100);
        
        // Subreddit name for the dataset
        const subredditName = "programming";
        
        // Create dashboard data object
        currentData = {
            totalPosts: mockPosts.length,
            totalComments: mockPosts.reduce((sum, post) => sum + post.num_comments, 0),
            uniqueAuthors: new Set(mockPosts.map(post => post.author)).size,
            totalScore: mockPosts.reduce((sum, post) => sum + post.score, 0),
            posts: mockPosts,
            subredditName: subredditName,
            timeData: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
                values: [12, 19, 15, 25, 32, 28, 21, 35, 29]
            },
            sentimentData: {
                labels: ['Positive', 'Negative', 'Neutral'],
                values: [45, 25, 30],
                colors: ['#43a047', '#e53935', '#757575']
            },
            keywordData: {
                labels: ['javascript', 'python', 'react', 'node.js', 'typescript', 'api', 'mongo', 'database', 'frontend', 'backend'],
                values: [85, 75, 65, 60, 55, 50, 45, 40, 35, 30]
            },
            topicData: {
                labels: ['Web Development', 'Data Science', 'Mobile Apps', 'DevOps', 'Security'],
                values: [35, 25, 20, 15, 5],
                colors: ['#1e88e5', '#43a047', '#7e57c2', '#ffa000', '#e53935']
            },
            subredditData: {
                labels: ['r/programming', 'r/javascript', 'r/python', 'r/webdev', 'r/reactjs', 'r/node', 'r/coding'],
                values: [30, 25, 20, 15, 5, 3, 2],
                colors: [
                    '#1e88e5', '#43a047', '#7e57c2', '#ffa000', 
                    '#e53935', '#00acc1', '#3949ab'
                ]
            }
        };
        
        // Update dashboard with mock data
        updateDashboard(currentData);
    }
    
    /**
     * Generate mock posts for demonstration
     * @param {Number} count - Number of posts to generate
     * @returns {Array} Array of mock post objects
     */
    function generateMockPosts(count) {
        const posts = [];
        const subreddits = ['programming', 'javascript', 'python', 'webdev', 'reactjs', 'node', 'coding'];
        
        const titles = [
            "How to optimize React performance in large applications",
            "What's your favorite VS Code extension for web development?",
            "Python vs JavaScript: Which should I learn first?",
            "Best practices for API design in 2023",
            "Dealing with technical debt in a startup environment",
            "TypeScript is changing the way I write JavaScript - here's why",
            "Modern authentication methods: JWT vs Session-based",
            "How I built a real-time chat application with WebSockets",
            "Lessons learned from scaling a database to handle millions of users",
            "Why functional programming concepts are important for all developers",
            "Comparing MongoDB and PostgreSQL for web applications",
            "The most underrated Node.js libraries you should know about",
            "Frontend state management in 2023: Redux vs Context API vs Zustand",
            "How Git changed the way we approach software development",
            "Microservices vs Monoliths: What's right for your project?",
            "Containerization with Docker: A beginner's guide",
            "Testing strategies that actually work in production",
            "Making sense of CSS Grid and Flexbox layouts",
            "Python automation scripts that saved me hours of work",
            "The future of web development: WebAssembly and beyond"
        ];
        
        const selfTexts = [
            "I've been working on optimizing our React application, and wanted to share some techniques that helped us reduce load times by 60%...",
            "After spending years in the industry, I've learned that robust error handling is the key difference between amateur and professional code...",
            "I recently switched from using REST APIs to GraphQL and wanted to share my experience and the lessons I learned along the way...",
            "Security is often overlooked in development. Here are some basic principles every developer should follow to protect user data...",
            "Working with legacy code can be challenging, but I've developed a systematic approach that makes it more manageable...",
            "The key to successful testing is not just writing tests, but writing the right kinds of tests that give you confidence in your code...",
            "Performance optimization isn't just about speed - it's about user experience. Here's how we improved our application's perceived performance...",
            "I've been experimenting with different state management libraries, and here's my analysis of the pros and cons of each approach...",
            "Database design decisions can make or break your application. These are the most important principles I follow when designing schemas...",
            "Accessibility should be built into your development process from the beginning. Here's how we've integrated it into our workflow..."
        ];
        
        const authors = ['dev_ninja', 'code_wizard', 'bug_hunter', 'web_guru', 'tech_enthusiast', 'async_await', 'regex_master', 'database_pro', 'ui_designer', 'system_architect'];
        
        // Generate random posts
        for (let i = 0; i < count; i++) {
            const titleIndex = Math.floor(Math.random() * titles.length);
            const textIndex = Math.floor(Math.random() * selfTexts.length);
            const subredditIndex = Math.floor(Math.random() * subreddits.length);
            const authorIndex = Math.floor(Math.random() * authors.length);
            
            // Random creation date within the last month
            const now = new Date();
            const randomDaysAgo = Math.floor(Math.random() * 30);
            const createdDate = new Date(now.getTime() - (randomDaysAgo * 24 * 60 * 60 * 1000));
            const createdUtc = Math.floor(createdDate.getTime() / 1000);
            
            // Random scores and comments
            const score = Math.floor(Math.random() * 5000);
            const numComments = Math.floor(Math.random() * 500);
            
            // Generate sentiment
            let sentiment = null;
            const randomSentiment = Math.random();
            if (randomSentiment < 0.33) {
                sentiment = { label: 'positive', score: 0.7 + (Math.random() * 0.3) };
            } else if (randomSentiment < 0.66) {
                sentiment = { label: 'negative', score: 0.7 + (Math.random() * 0.3) };
            } else {
                sentiment = { label: 'neutral', score: 0.7 + (Math.random() * 0.3) };
            }
            
            // Create post object
            const post = {
                id: `t3_${Math.random().toString(36).substring(2, 10)}`,
                title: titles[titleIndex],
                selftext: selfTexts[textIndex],
                author: authors[authorIndex],
                subreddit: subreddits[subredditIndex],
                created_utc: createdUtc,
                score: score,
                num_comments: numComments,
                upvote_ratio: 0.7 + (Math.random() * 0.3),
                url: `https://reddit.com/r/${subreddits[subredditIndex]}/comments/${Math.random().toString(36).substring(2, 10)}`,
                permalink: `/r/${subreddits[subredditIndex]}/comments/${Math.random().toString(36).substring(2, 10)}`,
                is_self: true,
                sentiment: sentiment
            };
            
            posts.push(post);
        }
        
        return posts;
    }
    
    /**
     * Show loading state
     */
    function showLoading() {
        // Add loading overlay
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center bg-white bg-opacity-75';
        loadingOverlay.style.zIndex = '1000';
        loadingOverlay.id = 'loading-overlay';
        loadingOverlay.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status"></div>
                <p class="mt-2">Loading dashboard data...</p>
            </div>
        `;
        
        document.body.appendChild(loadingOverlay);
    }
    
    /**
     * Hide loading state
     */
    function hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.remove();
        }
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
}); 