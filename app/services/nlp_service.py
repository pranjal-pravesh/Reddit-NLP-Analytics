import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import emoji
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import io
import base64

from app.core.config import settings
from app.utils.caching import memory_cache
from app.utils.logging_config import logger

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Spacy for advanced NLP
import spacy

# Add KeyBERT for transformer-based keyword extraction
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False

class NLPService:
    """
    Service for NLP operations including:
    - Sentiment analysis using CardiffNLP's Twitter-RoBERTa
    - Keyword extraction (TF-IDF and TextRank)
    - Topic modeling (LDA)
    - Time-based post frequency histograms
    """
    
    def __init__(self):
        self.sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.sentiment_pipeline = None
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.initialize_sentiment_model()
        
        # Initialize spaCy model for NLP tasks
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("Downloading spaCy model 'en_core_web_sm'")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
            
        # Initialize KeyBERT if available
        self.keybert_model = None
        if KEYBERT_AVAILABLE:
            try:
                self.keybert_model = KeyBERT()
                logger.info("KeyBERT model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize KeyBERT model: {str(e)}")
    
    def initialize_sentiment_model(self):
        """Initialize the sentiment analysis model"""
        try:
            # Load the sentiment model in a more controlled way
            model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=model, 
                tokenizer=tokenizer,
                max_length=512,
                truncation=True
            )
            logger.info(f"Initialized sentiment model: {self.sentiment_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment model: {str(e)}")
            self.sentiment_pipeline = None
    
    @memory_cache(maxsize=1000, ttl=3600)
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using the CardiffNLP Twitter-RoBERTa model.
        Includes emoji support and basic text preprocessing.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with sentiment scores and label
        """
        if not self.sentiment_pipeline:
            self.initialize_sentiment_model()
            if not self.sentiment_pipeline:
                return {"label": "neutral", "score": 0.5}
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Handle empty text
        if not processed_text.strip():
            return {"label": "neutral", "score": 0.5}
        
        try:
            # Truncate extremely long texts to avoid tokenizer issues
            if len(processed_text) > 1024:
                processed_text = processed_text[:1024]
            
            # Run through sentiment model
            result = self.sentiment_pipeline(processed_text)[0]
            
            # Map label to standard format
            label_map = {
                "positive": "positive",
                "negative": "negative", 
                "neutral": "neutral"
            }
            
            return {
                "label": label_map.get(result["label"], result["label"]),
                "score": float(result["score"]),
                "positive": float(result["score"]) if result["label"] == "positive" else 0.0,
                "negative": float(result["score"]) if result["label"] == "negative" else 0.0,
                "neutral": float(result["score"]) if result["label"] == "neutral" else 0.0
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "neutral", "score": 0.5}
    
    @memory_cache(maxsize=500, ttl=1800)
    def extract_keywords(
        self, 
        texts: List[str], 
        method: str = "tfidf", 
        num_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract keywords from a list of texts.
        
        Args:
            texts: List of text strings
            method: Keyword extraction method ('tfidf', 'textrank', 'keybert' or 'hybrid')
            num_keywords: Number of keywords to return
            
        Returns:
            List of dicts with keywords and scores
        """
        # Input validation
        if not texts:
            logger.warning("Empty texts list provided to extract_keywords")
            return []
            
        # Handle case where texts might be a single item
        if not isinstance(texts, list):
            texts = [str(texts)]
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Handle empty texts after preprocessing
        processed_texts = [text if text.strip() else " placeholder " for text in processed_texts]
        
        if method == "keybert" and self.keybert_model is not None:
            return self._extract_keywords_keybert(processed_texts, num_keywords)
        elif method == "hybrid" and self.keybert_model is not None:
            # Hybrid method combines keybert with tfidf for better results
            keybert_results = self._extract_keywords_keybert(processed_texts, num_keywords * 2)
            tfidf_results = self._extract_keywords_tfidf(processed_texts, num_keywords * 2)
            return self._combine_keyword_results(keybert_results, tfidf_results, num_keywords)
        elif method == "tfidf":
            return self._extract_keywords_tfidf(processed_texts, num_keywords)
        elif method == "textrank":
            return self._extract_keywords_textrank(processed_texts, num_keywords)
        else:
            logger.warning(f"Unsupported keyword extraction method: {method}")
            return self._extract_keywords_tfidf(processed_texts, num_keywords)
    
    @memory_cache(maxsize=100, ttl=3600)
    def topic_modeling(
        self, 
        texts: List[str], 
        num_topics: int = 5,
        method: str = "lda"
    ) -> Dict[str, Any]:
        """
        Perform topic modeling on a collection of texts.
        
        Args:
            texts: List of text strings
            num_topics: Number of topics to identify
            method: Topic modeling method ('lda' or 'bertopic')
            
        Returns:
            Dict with topics and keywords
        """
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Filter out empty texts
        filtered_texts = [text for text in processed_texts if text.strip()]
        
        if not filtered_texts:
            return {"topics": [], "topic_distribution": []}
        
        if method == "lda":
            return self._topic_modeling_lda(filtered_texts, num_topics)
        elif method == "bertopic":
            return self._topic_modeling_bertopic(filtered_texts, num_topics)
        else:
            logger.warning(f"Unsupported topic modeling method: {method}")
            return self._topic_modeling_lda(filtered_texts, num_topics)
    
    def generate_time_histogram(
        self, 
        timestamps: List[float],
        interval: str = "day"
    ) -> Dict[str, Any]:
        """
        Generate a time-based histogram of posts.
        
        Args:
            timestamps: List of UTC timestamps
            interval: Time interval ('hour', 'day', 'week', 'month')
            
        Returns:
            Dict with histogram data and base64 encoded plot
        """
        if not timestamps:
            return {
                "counts": [],
                "labels": [], 
                "plot": None
            }
        
        # Convert timestamps to datetime objects
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Create DataFrame
        df = pd.DataFrame({'date': dates})
        
        # Set frequency based on interval
        freq_map = {
            'hour': 'H',
            'day': 'D',
            'week': 'W',
            'month': 'M'
        }
        freq = freq_map.get(interval, 'D')
        
        # Generate histogram data
        if interval == 'hour':
            df['period'] = df['date'].dt.floor('H')
            format_str = '%Y-%m-%d %H:00'
        elif interval == 'day':
            df['period'] = df['date'].dt.floor('D')
            format_str = '%Y-%m-%d'
        elif interval == 'week':
            df['period'] = df['date'].dt.to_period('W').dt.start_time
            format_str = '%Y-%m-%d'
        elif interval == 'month':
            df['period'] = df['date'].dt.to_period('M').dt.start_time
            format_str = '%Y-%m'
        
        # Count posts per period
        counts = df.groupby('period').size().reset_index(name='count')
        
        # Ensure continuous periods (fill gaps with zeros)
        if len(counts) > 1:
            full_range = pd.date_range(
                start=counts['period'].min(),
                end=counts['period'].max(),
                freq=freq
            )
            counts = counts.set_index('period').reindex(full_range, fill_value=0).reset_index()
            counts.rename(columns={'index': 'period'}, inplace=True)
        
        # Format labels
        labels = [d.strftime(format_str) for d in counts['period']]
        count_values = counts['count'].tolist()
        
        # Generate plot
        plt.figure(figsize=(10, 5))
        plt.bar(labels, count_values)
        plt.xticks(rotation=45)
        plt.xlabel(f'Time ({interval})')
        plt.ylabel('Number of Posts')
        plt.title(f'Post Frequency by {interval.capitalize()}')
        plt.tight_layout()
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return {
            "counts": count_values,
            "labels": labels,
            "plot": plot_base64
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP tasks"""
        if not text:
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Handle emojis
        text = emoji.demojize(text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep emojis converted to text
        text = re.sub(r'[^\w\s:_]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_keywords_tfidf(
        self, 
        texts: List[str], 
        num_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract keywords using TF-IDF"""
        # Ensure texts are strings, not nested lists
        if texts and isinstance(texts[0], list):
            logger.warning("Received nested list in texts parameter, flattening")
            texts = [str(item) for sublist in texts for item in sublist]
        
        # Convert any non-string items to strings
        texts = [str(text) if not isinstance(text, str) else text for text in texts]
        
        # Log the content being processed
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i} for keyword extraction, length: {len(text)}")
            logger.debug(f"Text sample: {text[:100]}...")
        
        # Ensure we have enough content to extract keywords
        processed_texts = []
        for text in texts:
            # Ensure minimum text length by repeating short texts
            if len(text.split()) < 3:
                logger.warning(f"Text too short for keyword extraction - padding text")
                # Repeat the text to give TF-IDF more to work with
                text = (text + " ") * 3
            processed_texts.append(text)
        
        # Add a baseline document if needed
        if len(processed_texts) < 2:
            logger.info("Adding baseline document for TF-IDF comparison")
            processed_texts.append("reddit post content placeholder text common words")
            
        # Create TF-IDF vectorizer with more permissive settings for short texts
        tfidf = TfidfVectorizer(
            max_df=0.95,
            min_df=1,  # Accept terms that appear in just one document
            max_features=200,
            stop_words=list(self.stopwords),
            ngram_range=(1, 2)  # Include bigrams which can work better for short texts
        )
        
        try:
            # Fit and transform texts
            tfidf_matrix = tfidf.fit_transform(processed_texts)
            feature_names = tfidf.get_feature_names_out()
            
            logger.info(f"TF-IDF found {len(feature_names)} features")
            
            # Get top keywords for each document
            results = []
            for i in range(len(texts)):
                # Skip the dummy document if we added one
                if i >= len(processed_texts) - (1 if len(processed_texts) > len(texts) else 0):
                    continue
                    
                # Get feature indices sorted by TF-IDF score
                doc_tfidf = tfidf_matrix[i].toarray()[0]
                sorted_indices = doc_tfidf.argsort()[::-1]
                
                # Get top keywords and scores
                top_indices = sorted_indices[:num_keywords]
                keywords = [feature_names[idx] for idx in top_indices if doc_tfidf[idx] > 0]
                scores = [float(doc_tfidf[idx]) for idx in top_indices if doc_tfidf[idx] > 0]
                
                # If we got no keywords, try to extract some basic ones
                if not keywords and len(processed_texts[i].split()) > 3:
                    logger.warning(f"TF-IDF failed to extract keywords for document {i}, using fallback method")
                    # Fallback: use most common non-stopwords
                    words = [word.lower() for word in processed_texts[i].split() 
                             if word.lower() not in self.stopwords and len(word) > 2]
                    word_counts = Counter(words).most_common(num_keywords)
                    if word_counts:
                        keywords = [word for word, _ in word_counts]
                        # Normalize scores
                        max_count = max([count for _, count in word_counts]) if word_counts else 1
                        scores = [float(count) / max_count for _, count in word_counts]
                
                logger.info(f"Document {i}: Found {len(keywords)} keywords")
                results.append({
                    "keywords": keywords,
                    "scores": scores
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in TF-IDF keyword extraction: {str(e)}", exc_info=True)
            return [{"keywords": [], "scores": []}] * len(texts)
    
    def _extract_keywords_textrank(
        self, 
        texts: List[str], 
        num_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract keywords using TextRank-inspired approach"""
        results = []
        
        for text in texts:
            try:
                # Tokenize into sentences and then words
                sentences = nltk.sent_tokenize(text)
                word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
                
                # Remove stopwords and short words
                word_tokens = [
                    [word.lower() for word in sentence if word.lower() not in self.stopwords and len(word) > 2]
                    for sentence in word_tokens
                ]
                
                # Flatten and count word occurrences
                all_words = [word for sentence in word_tokens for word in sentence]
                word_counts = Counter(all_words)
                
                # Get most common words
                top_words = word_counts.most_common(num_keywords)
                keywords = [word for word, count in top_words]
                scores = [float(count) / max(word_counts.values()) for word, count in top_words]
                
                results.append({
                    "keywords": keywords,
                    "scores": scores
                })
            except Exception as e:
                logger.error(f"Error in TextRank keyword extraction: {str(e)}")
                results.append({"keywords": [], "scores": []})
        
        return results
    
    def _topic_modeling_lda(
        self, 
        texts: List[str], 
        num_topics: int = 5
    ) -> Dict[str, Any]:
        """Perform topic modeling using LDA"""
        try:
            # Create document-term matrix using Gensim
            tokens = [nltk.word_tokenize(text.lower()) for text in texts]
            
            # Filter out stopwords and short words
            filtered_tokens = [
                [word for word in doc if word not in self.stopwords and len(word) > 2]
                for doc in tokens
            ]
            
            # Create dictionary
            dictionary = corpora.Dictionary(filtered_tokens)
            
            # Filter out extreme values
            dictionary.filter_extremes(no_below=2, no_above=0.9)
            
            # Create document-term matrix
            corpus = [dictionary.doc2bow(doc) for doc in filtered_tokens]
            
            # Train LDA model
            lda_model = LdaModel(
                corpus=corpus, 
                id2word=dictionary, 
                num_topics=num_topics, 
                passes=15, 
                alpha='auto'
            )
            
            # Extract topics
            topics = []
            for topic_id in range(num_topics):
                topic_words = lda_model.show_topic(topic_id, topn=10)
                topics.append({
                    "id": topic_id,
                    "words": [word for word, prob in topic_words],
                    "probabilities": [float(prob) for word, prob in topic_words]
                })
            
            # Get document-topic distribution
            doc_topics = []
            for doc in corpus:
                topic_dist = lda_model.get_document_topics(doc)
                doc_topics.append({
                    "topic_ids": [topic_id for topic_id, prob in topic_dist],
                    "probabilities": [float(prob) for topic_id, prob in topic_dist]
                })
            
            return {
                "topics": topics,
                "topic_distribution": doc_topics
            }
        except Exception as e:
            logger.error(f"Error in LDA topic modeling: {str(e)}")
            return {"topics": [], "topic_distribution": []}
    
    def _topic_modeling_bertopic(
        self, 
        texts: List[str], 
        num_topics: int = 5
    ) -> Dict[str, Any]:
        """
        Perform topic modeling using BERTopic.
        This is optional and only runs if BERTopic is installed.
        """
        try:
            # Import BERTopic (it's optional, so we import it here)
            from bertopic import BERTopic
            
            # Initialize and fit model with reduced components for efficiency
            topic_model = BERTopic(
                nr_topics=num_topics,
                calculate_probabilities=True
            )
            
            # Fit model (using min_df to filter out rare words)
            topics, probs = topic_model.fit_transform(texts)
            
            # Extract topics and format results
            topic_results = []
            for topic_id in range(-1, min(num_topics, max(topics) + 1)):
                # -1 is always the outlier topic in BERTopic
                if topic_id == -1 and topic_id not in topics:
                    continue
                    
                if topic_id in topic_model.get_topic_info()["Topic"].values:
                    topic_words = topic_model.get_topic(topic_id)
                    topic_results.append({
                        "id": int(topic_id),
                        "words": [word for word, score in topic_words[:10]],
                        "probabilities": [float(score) for word, score in topic_words[:10]]
                    })
            
            # Get document-topic distribution
            doc_topics = []
            for i, doc_topics_vector in enumerate(probs):
                # Get probabilities for each topic
                topic_ids = [int(topic) for topic in sorted(range(len(doc_topics_vector)), 
                                                      key=lambda x: doc_topics_vector[x], 
                                                      reverse=True)[:5]]
                topic_probs = [float(doc_topics_vector[topic]) for topic in topic_ids]
                
                doc_topics.append({
                    "topic_ids": topic_ids,
                    "probabilities": topic_probs
                })
            
            return {
                "topics": topic_results,
                "topic_distribution": doc_topics
            }
            
        except ImportError:
            logger.warning("BERTopic not available, falling back to LDA")
            return self._topic_modeling_lda(texts, num_topics)
        except Exception as e:
            logger.error(f"Error in BERTopic topic modeling: {str(e)}")
            return {"topics": [], "topic_distribution": []}
    
    def _extract_keywords_keybert(
        self, 
        texts: List[str], 
        num_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """Extract keywords using KeyBERT and transformer models"""
        if not self.keybert_model:
            logger.warning("KeyBERT model not available, falling back to TF-IDF")
            return self._extract_keywords_tfidf(texts, num_keywords)
            
        try:
            results = []
            
            for text in texts:
                # Use keyphrase_ngram_range to control minimum and maximum length of keyphrases
                # Using multiple keyphrase_ngram_ranges to find the best keyphrases
                try:
                    keybert_keywords = self.keybert_model.extract_keywords(
                        text,
                        keyphrase_ngram_range=(1, 3),
                        stop_words='english',
                        use_mmr=True,  # Use Maximal Marginal Relevance to diversify results
                        diversity=0.7,  # Higher diversity for varied results
                        top_n=num_keywords
                    )
                    
                    if not keybert_keywords:
                        # Try again with different settings if no results
                        keybert_keywords = self.keybert_model.extract_keywords(
                            text,
                            keyphrase_ngram_range=(1, 2),
                            stop_words='english',
                            top_n=num_keywords
                        )
                        
                    # Extract keywords and scores from results
                    keywords = [kw[0] for kw in keybert_keywords]
                    scores = [float(kw[1]) for kw in keybert_keywords]
                    
                    # If still no keywords, use fallback method
                    if not keywords:
                        logger.warning(f"KeyBERT failed to extract keywords, using fallback method")
                        words = [word.lower() for word in text.split() 
                                if word.lower() not in self.stopwords and len(word) > 2]
                        word_counts = Counter(words).most_common(num_keywords)
                        if word_counts:
                            keywords = [word for word, _ in word_counts]
                            max_count = max([count for _, count in word_counts]) if word_counts else 1
                            scores = [float(count) / max_count for _, count in word_counts]
                    
                    results.append({
                        "keywords": keywords,
                        "scores": scores
                    })
                    
                except Exception as e:
                    logger.error(f"Error in KeyBERT keyword extraction for text: {str(e)}")
                    # Return empty result for this text
                    results.append({"keywords": [], "scores": []})
                
            return results
            
        except Exception as e:
            logger.error(f"Error in KeyBERT keyword extraction: {str(e)}", exc_info=True)
            # Fall back to TF-IDF
            return self._extract_keywords_tfidf(texts, num_keywords)
            
    def _combine_keyword_results(
        self,
        keybert_results: List[Dict[str, Any]],
        tfidf_results: List[Dict[str, Any]],
        num_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """Combine results from multiple keyword extraction methods"""
        # For each document, combine the results
        combined_results = []
        
        for i in range(min(len(keybert_results), len(tfidf_results))):
            # Get keywords and scores from both methods
            kb_keywords = keybert_results[i]["keywords"]
            kb_scores = keybert_results[i]["scores"]
            
            tf_keywords = tfidf_results[i]["keywords"]
            tf_scores = tfidf_results[i]["scores"]
            
            # Combine and deduplicate
            keyword_map = {}
            
            # Add KeyBERT keywords with higher weight
            for keyword, score in zip(kb_keywords, kb_scores):
                keyword_map[keyword] = score * 1.2  # Weight KeyBERT higher
                
            # Add TF-IDF keywords
            for keyword, score in zip(tf_keywords, tf_scores):
                if keyword in keyword_map:
                    # If keyword exists in both, take max
                    keyword_map[keyword] = max(keyword_map[keyword], score)
                else:
                    keyword_map[keyword] = score
                    
            # Sort by score and take top N
            sorted_keywords = sorted(keyword_map.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
            
            # Create result dict
            combined_results.append({
                "keywords": [k for k, s in sorted_keywords],
                "scores": [s for k, s in sorted_keywords]
            })
            
        return combined_results


# Create global instance
nlp_service = NLPService() 