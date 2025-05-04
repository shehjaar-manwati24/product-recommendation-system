import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple

class ContentBasedRecommender:
    """
    Content-based recommender using nearest neighbors on item features.
    """
    
    def __init__(self, n_neighbors: int = 10):
        self.item_features = None
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        
    def fit(self, item_features: np.ndarray) -> None:
        """
        Fit the content-based recommender using nearest neighbors.
        
        Args:
            item_features: Feature matrix for items
        """
        self.item_features = item_features
        self.nn_model.fit(item_features)
    
    def get_similar_items(self, item_id: int, 
                         n_items: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-N similar items for a given item.
        
        Args:
            item_id: Item ID to find similar items for
            n_items: Number of similar items to return
            
        Returns:
            Tuple of (similar item indices, similarity scores)
        """
        # Get the item features
        item_features = self.item_features[item_id].reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(item_features, n_neighbors=n_items+1)
        
        # Convert distances to similarities and exclude the item itself
        similarities = 1 - distances[0, 1:]  # Convert distance to similarity
        similar_items = indices[0, 1:]  # Exclude the item itself
        
        return similar_items, similarities
    
    def get_recommendations(self, item_ids: List[int],
                          n_items: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-N recommendations based on multiple items.
        
        Args:
            item_ids: List of item IDs to base recommendations on
            n_items: Number of items to recommend
            
        Returns:
            Tuple of (recommended item indices, recommendation scores)
        """
        # Get features for all input items
        input_features = self.item_features[item_ids]
        
        # Find nearest neighbors for each input item
        distances, indices = self.nn_model.kneighbors(input_features, n_neighbors=n_items+1)
        
        # Convert distances to similarities
        similarities = 1 - distances
        
        # Average similarities across input items
        avg_similarities = np.mean(similarities, axis=0)
        
        # Get unique recommended items and their scores
        recommended_items = np.unique(indices.flatten())
        recommendation_scores = np.zeros(len(recommended_items))
        
        for i, item in enumerate(recommended_items):
            # Average similarity scores for this item across all input items
            item_scores = similarities[np.where(indices == item)]
            recommendation_scores[i] = np.mean(item_scores)
        
        # Sort by score and return top n_items
        sort_idx = np.argsort(-recommendation_scores)
        return recommended_items[sort_idx][:n_items], recommendation_scores[sort_idx][:n_items]