# Hybrid Product Recommendation System

A modular, production-ready hybrid recommendation system that combines collaborative filtering and content-based approaches.

## Features

- Collaborative filtering using LightFM with WARP loss
- Content-based recommendations using cosine similarity
- Hybrid approach combining both methods
- FastAPI-based REST API for serving recommendations
- Evaluation metrics (Precision@K, Recall@K, MAP)

## Project Structure

```
.
├── app.py                 # FastAPI application
├── data_processor.py      # Data loading and preprocessing
├── collaborative_filtering.py  # Collaborative filtering model
├── content_based.py       # Content-based recommender
├── hybrid_recommender.py  # Hybrid recommender
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hybrid-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Prepare your data in CSV format:
   - Interactions data: user_id, item_id, timestamp
   - Item metadata: item_id, category, brand, title, description

2. Place your data files in the data directory

### Training the Model

```python
from data_processor import DataProcessor
from hybrid_recommender import HybridRecommender

# Initialize components
data_processor = DataProcessor()
recommender = HybridRecommender()

# Load and preprocess data
interactions_df, items_df = data_processor.load_data(
    'data/interactions.csv',
    'data/items.csv'
)

# Preprocess interactions
user_ids, item_ids, ratings = data_processor.preprocess_interactions(interactions_df)
interactions = data_processor.get_user_item_matrix(user_ids, item_ids, ratings)

# Preprocess item features
item_features = data_processor.preprocess_item_features(items_df)

# Train the model
recommender.fit(
    interactions=interactions,
    item_features=item_features,
    user_ids=user_ids,
    item_ids=item_ids,
    ratings=ratings,
    item_mapping=data_processor.item_mapping,
    user_mapping=data_processor.user_mapping
)

# Save the model
recommender.save_models('models/cf_model.pkl', 'models/cb_model.pkl')
```

### Running the API

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /`: Health check endpoint
- `POST /recommend`: Get recommendations for a user
  ```json
  {
    "user_id": 123,
    "n_items": 10
  }
  ```

## Evaluation

The system includes evaluation metrics:
- Precision@K
- Recall@K
- Mean Average Precision (MAP)

