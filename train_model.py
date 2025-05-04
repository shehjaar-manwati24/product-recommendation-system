from data_processor import DataProcessor
from hybrid_recommender import HybridRecommender
import os

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize components
    data_processor = DataProcessor()
    recommender = HybridRecommender()
    
    # Load and preprocess data
    print("Loading data...")
    interactions_df, items_df = data_processor.load_data(
        'data/interactions.csv',
        'data/items.csv'
    )
    
    # Preprocess interactions
    print("Preprocessing interactions...")
    user_ids, item_ids, ratings = data_processor.preprocess_interactions(interactions_df)
    interactions = data_processor.get_user_item_matrix(user_ids, item_ids, ratings)
    
    # Preprocess item features
    print("Preprocessing item features...")
    item_features = data_processor.preprocess_item_features(items_df)
    
    # Train the model
    print("Training model...")
    recommender.fit(
        interactions=interactions,
        item_features=item_features,
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        item_mapping=data_processor.item_mapping,
        user_mapping=data_processor.user_mapping
    )
    
    # Save the model and mappings
    print("Saving model...")
    recommender.save_models('models/cf_model.pkl', 'models/cb_model.pkl')
    
    # Save mappings
    import joblib
    joblib.dump(data_processor.item_mapping, 'models/item_mapping.pkl')
    joblib.dump(data_processor.user_mapping, 'models/user_mapping.pkl')
    
    print("Training complete! Models saved to 'models' directory.")

if __name__ == "__main__":
    main() 