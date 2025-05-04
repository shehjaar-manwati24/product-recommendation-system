import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List

class DataProcessor:
    """
    Handles data loading and preprocessing for the recommendation system.
    """
    
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_mapping = {}
        self.item_mapping = {}
        
    def load_data(self, interactions_path: str, items_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load interaction and item metadata data.
        """
        interactions_df = pd.read_csv(interactions_path)
        items_df = pd.read_csv(items_path)
        return interactions_df, items_df
    
    def preprocess_interactions(self, interactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess interaction data and create user-item matrix.
        """
        # Encode user and item IDs
        user_ids = self.user_encoder.fit_transform(interactions_df['user_id'])
        item_ids = self.item_encoder.fit_transform(interactions_df['item_id'])
        
        # Store mappings for later use
        self.user_mapping = dict(zip(interactions_df['user_id'], user_ids))
        self.item_mapping = dict(zip(interactions_df['item_id'], item_ids))
        
        # Create ratings array (assuming binary interactions)
        ratings = np.ones(len(interactions_df))
        
        return user_ids, item_ids, ratings
    
    def preprocess_item_features(self, items_df: pd.DataFrame, 
                               categorical_cols: List[str] = ['category', 'brand'],
                               text_cols: List[str] = ['title', 'description']) -> np.ndarray:
        """
        Preprocess item features for content-based recommendations.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import OneHotEncoder

        # Limit to top 20 categories and brands to avoid huge one-hot matrices
        top_categories = items_df['category'].value_counts().nlargest(20).index
        top_brands = items_df['brand'].value_counts().nlargest(20).index

        items_df['category'] = items_df['category'].where(items_df['category'].isin(top_categories), 'Other')
        items_df['brand'] = items_df['brand'].where(items_df['brand'].isin(top_brands), 'Other')

        # One-hot encode categorical features
        ohe = OneHotEncoder(sparse_output=False)
        categorical_features = ohe.fit_transform(items_df[categorical_cols])

        # Vectorize text features with a limit on the number of features
        tfidf = TfidfVectorizer(max_features=1000)
        text_features = tfidf.fit_transform(items_df[text_cols].fillna('').agg(' '.join, axis=1))

        # Combine features
        features = np.hstack([categorical_features, text_features.toarray()])
        
        return features
    
    def get_user_item_matrix(self, user_ids: np.ndarray, 
                           item_ids: np.ndarray, 
                           ratings: np.ndarray) -> np.ndarray:
        """
        Create a sparse user-item matrix from interactions.
        """
        from scipy.sparse import coo_matrix
        
        n_users = len(np.unique(user_ids))
        n_items = len(np.unique(item_ids))
        
        return coo_matrix((ratings, (user_ids, item_ids)), 
                         shape=(n_users, n_items)) 