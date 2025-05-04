import numpy as np
from typing import List, Tuple, Dict
from collaborative_filtering import CollaborativeFiltering
from content_based import ContentBasedRecommender

class HybridRecommender:
    """
    Hybrid recommender that combines collaborative filtering and content-based approaches.
    """
    
    def __init__(self, cf_weight: float = 0.5, cb_weight: float = 0.5):
        """
        Initialize the hybrid recommender.
        
        Args:
            cf_weight: Weight for collaborative filtering scores
            cb_weight: Weight for content-based scores
        """
        self.cf_model = CollaborativeFiltering()
        self.cb_model = ContentBasedRecommender()
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.item_mapping = {}
        self.user_mapping = {}
        
    def fit(self, interactions: np.ndarray,
            item_features: np.ndarray,
            user_ids: np.ndarray,
            item_ids: np.ndarray,
            ratings: np.ndarray,
            item_mapping: Dict,
            user_mapping: Dict) -> None:
        """
        Fit both collaborative filtering and content-based models.
        
        Args:
            interactions: User-item interaction matrix
            item_features: Feature matrix for items
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            ratings: Array of ratings/interactions
            item_mapping: Dictionary mapping original item IDs to encoded IDs
            user_mapping: Dictionary mapping original user IDs to encoded IDs
        """
        # Store mappings
        self.item_mapping = item_mapping
        self.user_mapping = user_mapping
        
        # Fit collaborative filtering model
        self.cf_model.train(interactions)
        
        # Fit content-based model
        self.cb_model.fit(item_features)
    
    def get_recommendations(self, user_id: int,
                          n_items: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hybrid recommendations for a given user.
        
        Args:
            user_id: User ID to generate recommendations for
            n_items: Number of items to recommend
            
        Returns:
            Tuple of (recommended item indices, recommendation scores)
        """
        # Get collaborative filtering recommendations
        cf_items = self.cf_model.get_recommendations(
            user_id,
            n_items=n_items
        )
        
        # Get content-based recommendations
        cb_items, cb_scores = self.cb_model.get_recommendations(
            [user_id],
            n_items=n_items
        )
        
        # Combine scores
        combined_scores = np.zeros(self.cf_model.interactions.shape[1])
        
        # Add collaborative filtering scores
        user_vector = self.cf_model.interactions[user_id].toarray().reshape(1, -1)
        distances, indices = self.cf_model.model.kneighbors(user_vector)
        similar_users = self.cf_model.interactions[indices[0]]
        cf_scores = similar_users.sum(axis=0).A1
        cf_scores[self.cf_model.interactions[user_id].nonzero()[1]] = 0
        combined_scores += self.cf_weight * cf_scores
        
        # Add content-based scores
        cb_scores_full = np.zeros_like(combined_scores)
        cb_scores_full[cb_items] = cb_scores
        combined_scores += self.cb_weight * cb_scores_full
        
        # Get top N items
        recommended_items = np.argsort(-combined_scores)[:n_items]
        recommendation_scores = combined_scores[recommended_items]
        
        return recommended_items, recommendation_scores
    
    def evaluate(self, train_interactions: np.ndarray,
                test_interactions: np.ndarray,
                k: int = 10) -> Tuple[float, float]:
        """
        Evaluate the hybrid model using precision@k and recall@k.
        
        Args:
            train_interactions: Training interaction matrix
            test_interactions: Test interaction matrix
            k: Number of items to consider for evaluation
            
        Returns:
            Tuple of (precision@k, recall@k)
        """
        return self.cf_model.evaluate(train_interactions, test_interactions, k)
    
    def save_models(self, cf_path: str, cb_path: str) -> None:
        """
        Save both models to disk.
        
        Args:
            cf_path: Path to save collaborative filtering model
            cb_path: Path to save content-based model
        """
        self.cf_model.save_model(cf_path)
        # Content-based model can be reconstructed from item features
    
    def load_models(self, cf_path: str) -> None:
        """
        Load collaborative filtering model from disk.
        
        Args:
            cf_path: Path to load collaborative filtering model from
        """
        self.cf_model.load_model(cf_path)