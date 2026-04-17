"""
FastAPI Recommendation Service
Provides REST API for e-commerce recommendation engine with <50ms latency
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from contextlib import asynccontextmanager
import time

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.collaborative_filtering import CollaborativeFilteringModel, create_spark_session
from models.deep_learning_embeddings import DeepLearningEmbeddingsModel, ColdStartHandler
from features.feature_store import EcommerceFeatureStore
from cache.redis_cache import RecommendationCache, CacheConfig
from ab_testing.ab_test_framework import ABTestFramework, MetricType, TestConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class UserRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    model_type: str = Field(default="hybrid", description="Model type: 'collaborative', 'deep_learning', or 'hybrid'")
    include_explanation: bool = Field(default=False, description="Include recommendation explanations")


class UserPreferences(BaseModel):
    categories: List[str] = Field(default=[], description="Preferred product categories")
    price_range: List[float] = Field(default=[0, 1000], description="Price range [min, max]")
    brands: List[str] = Field(default=[], description="Preferred brands")


class NewUserRequest(BaseModel):
    user_id: str = Field(..., description="New user identifier")
    preferences: UserPreferences = Field(..., description="User preferences for cold start")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")


class ItemRequest(BaseModel):
    item_id: str = Field(..., description="Item identifier")
    num_similar_items: int = Field(default=10, ge=1, le=50, description="Number of similar items")


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    model_used: str
    generated_at: datetime
    latency_ms: float
    cached: bool = False
    explanation: Optional[str] = None


class SimilarItemsResponse(BaseModel):
    item_id: str
    similar_items: List[Dict[str, Any]]
    generated_at: datetime
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    components: Dict[str, str]


# Global variables for services
spark_session = None
cf_model = None
dl_model = None
feature_store = None
cache = None
ab_framework = None
cold_start_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup services"""
    # Startup
    logger.info("Starting recommendation service...")
    
    try:
        # Initialize Spark session
        global spark_session, cf_model, dl_model, feature_store, cache, ab_framework, cold_start_handler
        
        spark_session = create_spark_session("RecommendationAPI")
        logger.info("Spark session created")
        
        # Initialize models
        cf_model = CollaborativeFilteringModel(spark_session)
        dl_model = DeepLearningEmbeddingsModel()
        cold_start_handler = ColdStartHandler()
        
        # Initialize feature store and cache
        feature_store = EcommerceFeatureStore()
        cache_config = CacheConfig()
        cache = RecommendationCache(cache_config)
        
        # Initialize A/B testing framework
        ab_framework = ABTestFramework()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down recommendation service...")
    if cache:
        cache.close()
    if spark_session:
        spark_session.stop()


# Create FastAPI app
app = FastAPI(
    title="E-commerce Recommendation API",
    description="High-performance recommendation engine for e-commerce",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def get_user_features_from_store(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user features from feature store or cache"""
    # Try cache first
    cached_features = cache.get_cached_user_features(user_id)
    if cached_features:
        return cached_features
    
    # Try feature store
    try:
        features_df = feature_store.get_user_features([user_id])
        if not features_df.empty:
            features = features_df.iloc[0].to_dict()
            cache.cache_user_features(user_id, features)
            return features
    except Exception as e:
        logger.error(f"Error getting user features: {e}")
    
    return None


def get_item_features_from_store(item_ids: List[str]) -> pd.DataFrame:
    """Get item features from feature store or cache"""
    features_list = []
    uncached_items = []
    
    # Try cache first
    for item_id in item_ids:
        cached_features = cache.get_cached_item_features(item_id)
        if cached_features:
            features_list.append(cached_features)
        else:
            uncached_items.append(item_id)
    
    # Get uncached items from feature store
    if uncached_items:
        try:
            features_df = feature_store.get_item_features(uncached_items)
            if not features_df.empty:
                for _, row in features_df.iterrows():
                    item_id = row['item_id']
                    features = row.to_dict()
                    cache.cache_item_features(item_id, features)
                    features_list.append(features)
        except Exception as e:
            logger.error(f"Error getting item features: {e}")
    
    return pd.DataFrame(features_list) if features_list else pd.DataFrame()


def generate_explanation(user_features: Dict, item_features: Dict, score: float) -> str:
    """Generate explanation for recommendation"""
    explanations = []
    
    # Category matching
    if user_features.get('favorite_category') and item_features.get('category'):
        if user_features['favorite_category'] == item_features['category']:
            explanations.append(f"Matches your interest in {item_features['category']}")
    
    # Price range
    if 'price_sensitivity' in user_features and 'price' in item_features:
        if user_features['price_sensitivity'] > 0.7 and item_features['price'] < 50:
            explanations.append("Good value for money")
        elif user_features['price_sensitivity'] < 0.3 and item_features['price'] > 200:
            explanations.append("Premium quality product")
    
    # Rating
    if item_features.get('avg_rating', 0) > 4.0:
        explanations.append(f"Highly rated ({item_features['avg_rating']:.1f}/5.0)")
    
    # Popularity
    if item_features.get('popularity_score', 0) > 10:
        explanations.append("Popular among customers")
    
    if not explanations:
        explanations.append("Recommended based on your preferences")
    
    return " | ".join(explanations)


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {}
    
    # Check Spark
    try:
        if spark_session:
            components["spark"] = "healthy"
        else:
            components["spark"] = "uninitialized"
    except Exception:
        components["spark"] = "error"
    
    # Check cache
    try:
        if cache and cache.health_check():
            components["cache"] = "healthy"
        else:
            components["cache"] = "error"
    except Exception:
        components["cache"] = "error"
    
    # Check feature store
    try:
        if feature_store:
            components["feature_store"] = "healthy"
        else:
            components["feature_store"] = "uninitialized"
    except Exception:
        components["feature_store"] = "error"
    
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(),
        components=components
    )


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: UserRequest, background_tasks: BackgroundTasks):
    """Get personalized recommendations for a user"""
    start_time = time.time()
    
    try:
        # Check cache first
        cached_recs = cache.get_cached_recommendations(request.user_id, request.model_type)
        if cached_recs:
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=cached_recs[:request.num_recommendations],
                model_used=request.model_type,
                generated_at=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000,
                cached=True
            )
        
        # Get user features
        user_features = get_user_features_from_store(request.user_id)
        
        if not user_features:
            # New user - handle cold start
            raise HTTPException(
                status_code=404,
                detail="User not found. Please use /recommendations/new-user endpoint for new users."
            )
        
        # Generate recommendations based on model type
        recommendations = []
        
        if request.model_type == "collaborative":
            # Use collaborative filtering
            if cf_model.model is None:
                raise HTTPException(status_code=503, detail="Collaborative filtering model not loaded")
            
            recs = cf_model.recommend_for_users([request.user_id], request.num_recommendations)
            if request.user_id in recs:
                for item_id, score in recs[request.user_id]:
                    recommendations.append({
                        "item_id": item_id,
                        "score": score,
                        "reason": "Based on similar users' preferences"
                    })
        
        elif request.model_type == "deep_learning":
            # Use deep learning model
            if dl_model.model is None:
                raise HTTPException(status_code=503, detail="Deep learning model not loaded")
            
            # Get candidate items (popular items for now)
            candidate_items = []  # This should come from your item catalog
            recs = dl_model.recommend_for_users([request.user_id], candidate_items, request.num_recommendations)
            if request.user_id in recs:
                for item_id, score in recs[request.user_id]:
                    recommendations.append({
                        "item_id": item_id,
                        "score": score,
                        "reason": "Based on neural network embeddings"
                    })
        
        else:  # hybrid
            # Combine both models (simple weighted average for now)
            all_recs = {}
            
            # Get collaborative filtering recommendations
            if cf_model.model is not None:
                cf_recs = cf_model.recommend_for_users([request.user_id], request.num_recommendations * 2)
                if request.user_id in cf_recs:
                    for item_id, score in cf_recs[request.user_id]:
                        all_recs[item_id] = all_recs.get(item_id, 0) + score * 0.6
            
            # Get deep learning recommendations
            if dl_model.model is not None:
                candidate_items = []  # Get from catalog
                dl_recs = dl_model.recommend_for_users([request.user_id], candidate_items, request.num_recommendations * 2)
                if request.user_id in dl_recs:
                    for item_id, score in dl_recs[request.user_id]:
                        all_recs[item_id] = all_recs.get(item_id, 0) + score * 0.4
            
            # Sort and format recommendations
            sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
            for item_id, score in sorted_recs[:request.num_recommendations]:
                recommendations.append({
                    "item_id": item_id,
                    "score": score,
                    "reason": "Hybrid recommendation combining multiple approaches"
                })
        
        # Get item features for explanations
        if request.include_explanation and recommendations:
            item_ids = [rec["item_id"] for rec in recommendations]
            item_features_df = get_item_features_from_store(item_ids)
            
            for i, rec in enumerate(recommendations):
                item_features = item_features_df[item_features_df['item_id'] == rec["item_id"]]
                if not item_features.empty:
                    item_features_dict = item_features.iloc[0].to_dict()
                    explanation = generate_explanation(user_features, item_features_dict, rec["score"])
                    recommendations[i]["explanation"] = explanation
        
        # Cache recommendations in background
        background_tasks.add_task(
            cache.cache_recommendations,
            request.user_id,
            recommendations,
            request.model_type
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used=request.model_type,
            generated_at=datetime.now(),
            latency_ms=latency_ms,
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/recommendations/new-user", response_model=RecommendationResponse)
async def get_new_user_recommendations(request: NewUserRequest):
    """Get recommendations for a new user using cold start"""
    start_time = time.time()
    
    try:
        # Create user profile
        cold_start_handler.create_user_profile(
            request.user_id,
            {
                "categories": request.preferences.categories,
                "price_range": request.preferences.price_range,
                "brands": request.preferences.brands
            }
        )
        
        # Get item catalog (this should come from your database)
        # For now, we'll return a placeholder
        item_catalog = pd.DataFrame()  # This should contain your item catalog
        
        # Get cold start recommendations
        recommendations_list = cold_start_handler.get_cold_start_recommendations(
            request.user_id,
            item_catalog,
            request.num_recommendations
        )
        
        recommendations = []
        for item_id, score in recommendations_list:
            recommendations.append({
                "item_id": item_id,
                "score": score,
                "reason": "Based on your preferences and popular items"
            })
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used="cold_start",
            generated_at=datetime.now(),
            latency_ms=latency_ms,
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Error generating cold start recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/items/similar", response_model=SimilarItemsResponse)
async def get_similar_items(request: ItemRequest):
    """Get similar items for a given item"""
    start_time = time.time()
    
    try:
        similar_items = []
        
        # Use deep learning model for similarity
        if dl_model.model is not None:
            # Get candidate items
            candidate_items = []  # This should come from your item catalog
            similar = dl_model.find_similar_items(request.item_id, candidate_items, request.num_similar_items)
            
            for item_id, similarity_score in similar:
                similar_items.append({
                    "item_id": item_id,
                    "similarity_score": similarity_score,
                    "reason": "Similar based on embeddings"
                })
        
        # Use collaborative filtering as fallback
        if not similar_items and cf_model.model is not None:
            similar_users = cf_model.recommend_for_items([request.item_id], request.num_similar_items)
            if request.item_id in similar_users:
                # Convert user similarity to item similarity (simplified)
                for user_id, score in similar_users[request.item_id]:
                    similar_items.append({
                        "item_id": f"similar_via_user_{user_id}",  # This needs proper implementation
                        "similarity_score": score,
                        "reason": "Users who liked this also liked"
                    })
        
        latency_ms = (time.time() - start_time) * 1000
        
        return SimilarItemsResponse(
            item_id=request.item_id,
            similar_items=similar_items[:request.num_similar_items],
            generated_at=datetime.now(),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Error finding similar items: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}/features")
async def get_user_features(user_id: str):
    """Get features for a specific user"""
    try:
        features = get_user_features_from_store(user_id)
        if not features:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"user_id": user_id, "features": features}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user features: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/items/{item_id}/features")
async def get_item_features(item_id: str):
    """Get features for a specific item"""
    try:
        features_df = get_item_features_from_store([item_id])
        if features_df.empty:
            raise HTTPException(status_code=404, detail="Item not found")
        
        features = features_df.iloc[0].to_dict()
        return {"item_id": item_id, "features": features}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting item features: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        if not cache:
            raise HTTPException(status_code=503, detail="Cache not available")
        
        stats = cache.get_cache_stats()
        return {"cache_stats": stats}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/cache/invalidate/user/{user_id}")
async def invalidate_user_cache(user_id: str):
    """Invalidate cache for a specific user"""
    try:
        if not cache:
            raise HTTPException(status_code=503, detail="Cache not available")
        
        success = cache.invalidate_user_cache(user_id)
        return {"user_id": user_id, "invalidated": success}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating user cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# A/B Testing endpoints
@app.post("/ab-tests")
async def create_ab_test(config: TestConfig):
    """Create a new A/B test"""
    try:
        test_id = ab_framework.create_test(config)
        return {"test_id": test_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/ab-tests/{test_id}/status")
async def get_ab_test_status(test_id: str):
    """Get A/B test status"""
    try:
        status = ab_framework.get_test_status(test_id)
        if not status:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ab-tests/{test_id}/start")
async def start_ab_test(test_id: str):
    """Start an A/B test"""
    try:
        success = ab_framework.start_test(test_id)
        if not success:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return {"test_id": test_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting A/B test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/ab-tests")
async def list_ab_tests():
    """List all A/B tests"""
    try:
        active_tests = ab_framework.list_active_tests()
        completed_tests = ab_framework.list_completed_tests()
        
        return {
            "active_tests": active_tests,
            "completed_tests": completed_tests
        }
        
    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
