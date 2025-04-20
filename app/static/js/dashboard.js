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
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [data.positive, data.negative, data.neutral],
                    backgroundColor: ['#43a047', '#e53935', '#757575'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
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
     * Initialize dashboard with mock data for demo purposes
     */
    function initializeMockData() {
        currentData = {
            totalPosts: 1254,
            totalComments: 8763,
            uniqueAuthors: 876,
            totalScore: 42589,
            
            timeData: {
                labels: ['Jan 1', 'Jan 2', 'Jan 3', 'Jan 4', 'Jan 5', 'Jan 6', 'Jan 7', 'Jan 8', 'Jan 9', 'Jan 10'],
                values: [45, 62, 81, 75, 92, 103, 87, 79, 94, 101]
            },
            
            sentimentData: {
                positive: 42,
                negative: 28,
                neutral: 30
            },
            
            keywordData: {
                labels: ['Python', 'JavaScript', 'React', 'Vue', 'Angular', 'Node.js', 'Django', 'Flask', 'ML', 'AI'],
                values: [78, 65, 59, 42, 35, 62, 48, 39, 55, 61]
            },
            
            topicData: {
                labels: ['Programming', 'Data Science', 'Web Dev', 'Mobile', 'AI/ML', 'DevOps'],
                values: [0.85, 0.72, 0.9, 0.65, 0.78, 0.63]
            },
            
            subredditData: {
                labels: ['r/programming', 'r/datascience', 'r/webdev', 'r/MachineLearning', 'r/javascript', 'r/Python'],
                values: [324, 281, 265, 198, 187, 145]
            },
            
            posts: generateMockPosts(50)
        };
        
        // Update dashboard
        updateDashboard(currentData);
    }
    
    /**
     * Generate mock posts for demo
     * @param {number} count - Number of posts to generate
     * @returns {Array} Array of post objects
     */
    function generateMockPosts(count) {
        const subreddits = ['programming', 'Python', 'javascript', 'webdev', 'datascience', 'MachineLearning'];
        const titles = [
            'Just built my first React app, what do you think?',
            'How to optimize Python code for better performance',
            'The future of AI in web development',
            'Tips for becoming a better programmer',
            'Why TypeScript is better than JavaScript',
            'Flask vs Django: Which one should you use?',
            'Machine Learning for beginners: Where to start',
            'What programming language should I learn in 2023?',
            'Building scalable web applications with Node.js',
            'Data visualization tools every data scientist should know'
        ];
        
        const posts = [];
        const now = Math.floor(Date.now() / 1000);
        
        for (let i = 0; i < count; i++) {
            const subreddit = subreddits[Math.floor(Math.random() * subreddits.length)];
            const title = titles[Math.floor(Math.random() * titles.length)];
            const score = Math.floor(Math.random() * 5000);
            const num_comments = Math.floor(Math.random() * 500);
            const created_utc = now - Math.floor(Math.random() * 30 * 24 * 60 * 60); // Random time in the last 30 days
            
            // Random sentiment
            const sentimentOptions = ['positive', 'negative', 'neutral'];
            const sentimentLabel = sentimentOptions[Math.floor(Math.random() * sentimentOptions.length)];
            const sentimentScore = Math.random() * 0.5 + 0.5; // Random score between 0.5 and 1.0
            
            posts.push({
                title,
                subreddit,
                author: `user${i + 1}`,
                score,
                num_comments,
                created_utc,
                url: '#',
                is_self: Math.random() > 0.5,
                sentiment: {
                    label: sentimentLabel,
                    score: sentimentScore
                }
            });
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