# Recommendation Systems with IBM Watson Studio

This project implements multiple recommendation algorithms using real data from the IBM Watson Studio platform. The goal is to create a comprehensive recommendation system that can suggest relevant articles to users based on different approaches and scenarios.

## Project Overview

The project explores and implements four different recommendation techniques:

1. **Rank-Based Recommendations** - Recommending the most popular articles
2. **User-User Based Collaborative Filtering** - Finding similar users and recommending articles they liked
3. **Content-Based Recommendations** - Recommending articles similar in content using NLP techniques
4. **Matrix Factorization** - Using SVD to find latent features for recommendations

## Dataset

The project uses user-article interaction data from IBM Watson Studio, containing:
- **45,993 user-article interactions**
- **5,149 unique users**
- **714 unique articles**
- Article titles and user engagement data

## Key Statistics

- 50% of users interact with **3 or fewer articles**
- Maximum interactions by a single user: **364 articles**
- Most popular article: **"Use Regression to Predict Iowa Liquor Sales"** (viewed 937 times)
- The dataset contains some users with missing email information (treated as "unknown_user")

## Implementation Details

### 1. Exploratory Data Analysis
- Data cleaning and preprocessing
- Statistical analysis of user-article interactions
- Visualization of interaction patterns
- Handling missing values in user data

### 2. Rank-Based Recommendations
```python
def get_top_articles(n, df=df):
    """Returns the top n most popular articles based on interaction count"""
```
- Simple approach based on article popularity
- Ideal for new users with no interaction history
- Recommends articles with the highest engagement

### 3. User-User Collaborative Filtering
```python
def user_user_recs(user_id, m=10):
    """Recommends articles based on similar users' preferences"""
```
- Uses cosine similarity to find similar users
- Creates user-item matrix (5149 users × 714 articles)
- Recommends articles that similar users have interacted with
- Implements tie-breaking using user activity levels

### 4. Content-Based Recommendations
```python
def make_content_recs(article_id, n, df=df):
    """Recommends articles with similar content using NLP clustering"""
```
- Uses **TF-IDF vectorization** on article titles
- Applies **Latent Semantic Analysis (LSA)** for dimensionality reduction
- Performs **K-Means clustering** (50 clusters) to group similar articles
- Recommends articles from the same content cluster

### 5. Matrix Factorization (SVD)
```python
def get_svd_similar_article_ids(article_id, vt, user_item=user_item):
    """Finds similar articles using SVD latent features"""
```
- Uses **Singular Value Decomposition** to find latent features
- Tested different numbers of components (10-700)
- Optimal performance around **200 latent features**
- Achieves high accuracy in predicting user-article interactions

## Technical Implementation

### Libraries Used
- **pandas** & **numpy** - Data manipulation and analysis
- **matplotlib** - Data visualization
- **scikit-learn** - Machine learning algorithms (TF-IDF, SVD, K-Means, Cosine Similarity)
- **sklearn.decomposition.TruncatedSVD** - Matrix factorization
- **sklearn.feature_extraction.text.TfidfVectorizer** - Text processing
- **sklearn.cluster.KMeans** - Content clustering

### Key Functions

#### Data Processing
- `email_mapper()` - Maps user emails to numerical IDs
- `create_user_item_matrix()` - Creates binary interaction matrix

#### Recommendation Functions
- `get_top_articles()` - Popularity-based recommendations
- `find_similar_users()` - User similarity calculation
- `user_user_recs()` - Collaborative filtering recommendations
- `make_content_recs()` - Content-based recommendations
- `get_svd_similar_article_ids()` - Matrix factorization recommendations

#### Utility Functions
- `get_article_names()` - Converts article IDs to titles
- `get_user_articles()` - Gets articles interacted with by a user
- `get_ranked_article_unique_counts()` - Ranks articles by popularity

## Model Performance

### Matrix Factorization Results
- **Accuracy**: Improves with more latent features, plateaus around 200 features
- **Precision & Recall**: Optimal balance achieved with 200 latent features
- Successfully reconstructs user-item interaction patterns

### Content-Based Clustering
- **50 clusters** determined via elbow method
- **200 max features** for TF-IDF
- **50 components** for LSA with explained variance of ~40%

## Recommendation Strategies by User Type

### New Users (No History)
- **Strategy**: Rank-based recommendations
- **Reason**: No interaction data available
- **Implementation**: Use `get_top_articles()` to recommend most popular content

### Users with Limited History
- **Strategy**: Hybrid approach (Rank-based + Content-based)
- **Reason**: Limited data for collaborative filtering
- **Implementation**: Combine popular articles with content similar to user's few interactions

### Users with Rich History
- **Strategy**: User-User Collaborative Filtering + Matrix Factorization
- **Reason**: Sufficient data for finding similar users and latent preferences
- **Implementation**: Use `user_user_recs()` and SVD-based recommendations

## Files Structure

```
├── Recommendations_with_IBM.ipynb    # Main notebook with all implementations
├── Recommendations_with_IBM.html     # HTML version of the notebook
├── project_tests.py                  # Test functions for validation
├── data/
│   └── user-item-interactions.csv    # Raw interaction data
├── top_5.p, top_10.p, top_20.p      # Cached recommendation results
└── README.md                         # This file
```

## Key Insights

1. **Cold Start Problem**: New users benefit most from popularity-based recommendations
2. **Sparsity**: The user-item matrix is highly sparse (most users interact with few articles)
3. **Content Clustering**: Article titles provide meaningful content similarity signals
4. **Matrix Factorization**: SVD effectively captures latent user preferences and article characteristics
5. **Scalability**: Different algorithms suit different scales of user activity

## Future Improvements

1. **Hybrid Models**: Combine multiple recommendation approaches
2. **Additional Features**: Incorporate article content beyond titles (abstracts, categories, tags)
3. **Advanced NLP**: Use word embeddings or transformer models for better content understanding
4. **Temporal Analysis**: Consider time-based patterns in user behavior
5. **Evaluation Metrics**: Implement proper offline evaluation using train/test splits
6. **Real-time Updates**: Develop incremental learning capabilities

## Usage

1. **Data Loading**: Load the user-item interaction data
2. **Preprocessing**: Clean data and create user-item matrix
3. **Model Training**: Fit the desired recommendation model
4. **Generate Recommendations**: Use appropriate function based on user type and scenario
5. **Evaluation**: Test recommendations and iterate improvements

## Conclusion

This project demonstrates a comprehensive approach to building recommendation systems, showcasing different algorithms suitable for various scenarios. The implementation covers the full pipeline from data exploration to model deployment, providing insights into the strengths and limitations of each approach.

The multi-faceted approach ensures robust recommendations across different user types and scenarios, from complete newcomers to power users with extensive interaction histories.