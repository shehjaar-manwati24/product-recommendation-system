import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
import joblib
from typing import Tuple, List

class CollaborativeFiltering:
    """
    Collaborative filtering recommender using k-nearest neighbors.
    """
    
    def __init__(self, n_neighbors: int = 20, metric: str = 'cosine'):
        """
        Initialize the collaborative filtering model.
        
        Args:
            n_neighbors: Number of neighbors to consider
            metric: Distance metric to use ('cosine', 'euclidean', etc.)
        """
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm='brute'
        )
        self.interactions = None
        
    def train(self, interactions: coo_matrix) -> None:
        """
        Train the collaborative filtering model.
        
        Args:
            interactions: User-item interaction matrix
        """
        self.interactions = interactions
        # Convert to CSR format for efficient row operations
        interactions_csr = interactions.tocsr()
        self.model.fit(interactions_csr)
    
    def get_recommendations(self, user_id: int, 
                          n_items: int = 10) -> np.ndarray:
        """
        Generate top-N recommendations for a given user.
        
        Args:
            user_id: User ID to generate recommendations for
            n_items: Number of items to recommend
            
        Returns:
            Array of recommended item indices
        """
        if self.interactions is None:
            raise ValueError("Model has not been trained yet")
            
        # Get user's interaction vector
        user_vector = self.interactions[user_id].toarray().reshape(1, -1)
        
        # Find similar users
        distances, indices = self.model.kneighbors(user_vector)
        
        # Get items from similar users
        similar_users = self.interactions[indices[0]]
        
        # Aggregate interactions from similar users
        item_scores = similar_users.sum(axis=0).A1
        
        # Remove items the user has already interacted with
        item_scores[self.interactions[user_id].nonzero()[1]] = 0
        
        # Get top N items
        top_items = np.argsort(-item_scores)[:n_items]
        
        return top_items
    
    def evaluate(self, train_interactions: coo_matrix,
                test_interactions: coo_matrix,
                k: int = 10) -> Tuple[float, float]:
        """
        Evaluate the model using precision@k and recall@k.
        
        Args:
            train_interactions: Training interaction matrix
            test_interactions: Test interaction matrix
            k: Number of items to consider for evaluation
            
        Returns:
            Tuple of (precision@k, recall@k)
        """
        precision_scores = []
        recall_scores = []
        
        # Convert to CSR format for efficient row operations
        test_interactions_csr = test_interactions.tocsr()
        
        for user_id in range(test_interactions.shape[0]):
            # Get ground truth items
            ground_truth = set(test_interactions_csr[user_id].nonzero()[1])
            
            if len(ground_truth) == 0:
                continue
                
            # Get recommendations
            recommended = set(self.get_recommendations(user_id, k))
            
            # Calculate precision and recall
            if len(recommended) > 0:
                precision = len(ground_truth & recommended) / len(recommended)
                recall = len(ground_truth & recommended) / len(ground_truth)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
        
        return np.mean(precision_scores), np.mean(recall_scores)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump({
            'model': self.model,
            'interactions': self.interactions
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.interactions = saved_data['interactions'] 