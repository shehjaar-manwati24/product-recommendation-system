from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from data_processor import DataProcessor
from hybrid_recommender import HybridRecommender
import joblib

app = FastAPI(title="Hybrid Recommender System")

# Initialize components
data_processor = DataProcessor()
recommender = HybridRecommender()

# Load data and models (to be implemented)
# This would typically be done in a startup event or similar
try:
    # Load mappings and models
    with open('models/item_mapping.pkl', 'rb') as f:
        item_mapping = joblib.load(f)
    with open('models/user_mapping.pkl', 'rb') as f:
        user_mapping = joblib.load(f)
    
    # Load recommender model
    recommender.load_models('models/cf_model.pkl')
except FileNotFoundError:
    print("Models not found. Please train the models first.")

class RecommendationRequest(BaseModel):
    user_id: int
    n_items: Optional[int] = 10

class RecommendationResponse(BaseModel):
    item_ids: List[int]
    scores: List[float]

@app.get("/")
async def root():
    return {"message": "Hybrid Recommender System API"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get recommendations for a given user.
    
    Args:
        request: RecommendationRequest containing user_id and optional n_items
        
    Returns:
        RecommendationResponse containing recommended item IDs and scores
    """
    try:
        # Convert user ID to internal representation
        internal_user_id = recommender.user_mapping.get(request.user_id)
        if internal_user_id is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get recommendations
        items, scores = recommender.get_recommendations(
            internal_user_id,
            n_items=request.n_items
        )
        
        # Convert item IDs back to original format
        original_items = [k for k, v in recommender.item_mapping.items() if v in items]
        
        return RecommendationResponse(
            item_ids=original_items,
            scores=scores.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 