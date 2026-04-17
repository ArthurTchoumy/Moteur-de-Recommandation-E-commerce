"""
Feature Store implementation using Feast
For e-commerce recommendation engine
"""

from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.data_source import PushSource
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcommerceFeatureStore:
    """
    Feature Store for e-commerce recommendation system
    Manages user and item features for real-time serving
    """
    
    def __init__(self, repo_path: str = "feature_repo"):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(exist_ok=True)
        self.store = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize Feast feature store"""
        try:
            self.store = FeatureStore(repo_path=str(self.repo_path))
            logger.info("Feature store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize feature store: {e}")
            self._create_feature_store()
    
    def _create_feature_store(self):
        """Create feature store configuration"""
        # Define entities
        user_entity = Entity(
            name="user_id",
            join_keys=["user_id"],
            description="User identifier"
        )
        
        item_entity = Entity(
            name="item_id", 
            join_keys=["item_id"],
            description="Item identifier"
        )
        
        # Define data sources
        user_source = FileSource(
            path=str(self.repo_path / "user_features.parquet"),
            timestamp_field="event_timestamp",
            created_timestamp_column="created_timestamp"
        )
        
        item_source = FileSource(
            path=str(self.repo_path / "item_features.parquet"),
            timestamp_field="event_timestamp",
            created_timestamp_column="created_timestamp"
        )
        
        # Define feature views
        user_features = FeatureView(
            name="user_features",
            entities=["user_id"],
            ttl=timedelta(days=30),
            schema=[
                Field(name="avg_rating", dtype=Float32),
                Field(name="rating_count", dtype=Int64),
                Field(name="favorite_category", dtype=String),
                Field(name="price_sensitivity", dtype=Float32),
                Field(name="activity_level", dtype=Float32),
                Field(name="account_age_days", dtype=Int64),
                Field(name="last_purchase_days_ago", dtype=Int64),
            ],
            source=user_source
        )
        
        item_features = FeatureView(
            name="item_features",
            entities=["item_id"],
            ttl=timedelta(days=7),
            schema=[
                Field(name="category", dtype=String),
                Field(name="brand", dtype=String),
                Field(name="price", dtype=Float32),
                Field(name="avg_rating", dtype=Float32),
                Field(name="review_count", dtype=Int64),
                Field(name="popularity_score", dtype=Float32),
                Field(name="availability", dtype=String),
                Field(name="discount_percentage", dtype=Float32),
            ],
            source=item_source
        )
        
        # Create feature store registry
        registry_content = f"""
project: ecommerce_features
registry: {str(self.repo_path / "registry.db")}
provider: local
offline_store:
    type: file
online_store:
    type: redis
    connection_string: "localhost:6379"
"""
        
        with open(self.repo_path / "feature_store.yaml", "w") as f:
            f.write(registry_content)
        
        # Apply feature store
        self.store = FeatureStore(repo_path=str(self.repo_path))
        self.store.apply([user_entity, item_entity, user_features, item_features])
        logger.info("Feature store created and applied")
    
    def prepare_user_features(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare user features from interaction and user data
        """
        logger.info("Preparing user features")
        
        # Calculate user statistics from interactions
        user_stats = interactions_df.groupby('user_id').agg({
            'rating': ['mean', 'count'],
            'timestamp': ['min', 'max'],
            'price': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        user_stats.columns = ['user_id', 'avg_rating', 'rating_count', 
                            'first_interaction', 'last_interaction', 
                            'avg_price', 'std_price']
        
        # Calculate additional features
        user_stats['favorite_category'] = interactions_df.groupby('user_id')['category']
        user_stats['favorite_category'] = user_stats['favorite_category'].apply(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'unknown'
        )
        
        # Price sensitivity (inverse of price variance)
        user_stats['price_sensitivity'] = 1 / (user_stats['std_price'] + 1)
        
        # Activity level (interactions per day)
        user_stats['activity_level'] = user_stats['rating_count'] / (
            (user_stats['last_interaction'] - user_stats['first_interaction']).dt.days + 1
        )
        
        # Account age (days since first interaction)
        current_time = pd.Timestamp.now()
        user_stats['account_age_days'] = (current_time - user_stats['first_interaction']).dt.days
        
        # Days since last purchase
        user_stats['last_purchase_days_ago'] = (current_time - user_stats['last_interaction']).dt.days
        
        # Add timestamp for Feast
        user_stats['event_timestamp'] = current_time
        user_stats['created_timestamp'] = current_time
        
        # Select relevant columns
        feature_columns = [
            'user_id', 'avg_rating', 'rating_count', 'favorite_category',
            'price_sensitivity', 'activity_level', 'account_age_days',
            'last_purchase_days_ago', 'event_timestamp', 'created_timestamp'
        ]
        
        return user_stats[feature_columns]
    
    def prepare_item_features(self, items_df: pd.DataFrame, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare item features from item and interaction data
        """
        logger.info("Preparing item features")
        
        # Calculate item statistics from interactions
        item_stats = interactions_df.groupby('item_id').agg({
            'rating': ['mean', 'count'],
            'user_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        item_stats.columns = ['item_id', 'avg_rating', 'review_count', 'unique_users']
        
        # Calculate popularity score (weighted by rating and review count)
        item_stats['popularity_score'] = (
            item_stats['avg_rating'] * np.log1p(item_stats['review_count'])
        )
        
        # Merge with item metadata
        item_features = items_df.merge(item_stats, on='item_id', how='left')
        
        # Fill missing values for new items
        item_features['avg_rating'] = item_features['avg_rating'].fillna(0.0)
        item_features['review_count'] = item_features['review_count'].fillna(0)
        item_features['popularity_score'] = item_features['popularity_score'].fillna(0.0)
        item_features['unique_users'] = item_features['unique_users'].fillna(0)
        
        # Add timestamp for Feast
        current_time = pd.Timestamp.now()
        item_features['event_timestamp'] = current_time
        item_features['created_timestamp'] = current_time
        
        # Select relevant columns
        feature_columns = [
            'item_id', 'category', 'brand', 'price', 'avg_rating', 
            'review_count', 'popularity_score', 'availability', 
            'discount_percentage', 'event_timestamp', 'created_timestamp'
        ]
        
        return item_features[feature_columns]
    
    def ingest_features(self, user_features_df: pd.DataFrame, item_features_df: pd.DataFrame):
        """
        Ingest features into the feature store
        """
        logger.info("Ingesting features into feature store")
        
        # Save features to parquet files
        user_features_path = self.repo_path / "user_features.parquet"
        item_features_path = self.repo_path / "item_features.parquet"
        
        user_features_df.to_parquet(user_features_path, index=False)
        item_features_df.to_parquet(item_features_path, index=False)
        
        # Materialize features
        self.store.materialize_incremental(end_date=datetime.now())
        
        logger.info("Features ingested successfully")
    
    def get_user_features(self, user_ids: List[str]) -> pd.DataFrame:
        """
        Retrieve features for specific users
        """
        try:
            feature_service = self.store.get_feature_service(
                "user_features_service",
                features=[
                    "user_features:avg_rating",
                    "user_features:rating_count", 
                    "user_features:favorite_category",
                    "user_features:price_sensitivity",
                    "user_features:activity_level",
                    "user_features:account_age_days",
                    "user_features:last_purchase_days_ago"
                ]
            )
            
            # Create entity DataFrame
            entity_df = pd.DataFrame({
                "user_id": user_ids,
                "event_timestamp": [datetime.now()] * len(user_ids)
            })
            
            # Retrieve features
            features = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_service
            ).to_df()
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve user features: {e}")
            return pd.DataFrame()
    
    def get_item_features(self, item_ids: List[str]) -> pd.DataFrame:
        """
        Retrieve features for specific items
        """
        try:
            feature_service = self.store.get_feature_service(
                "item_features_service",
                features=[
                    "item_features:category",
                    "item_features:brand",
                    "item_features:price",
                    "item_features:avg_rating",
                    "item_features:review_count",
                    "item_features:popularity_score",
                    "item_features:availability",
                    "item_features:discount_percentage"
                ]
            )
            
            # Create entity DataFrame
            entity_df = pd.DataFrame({
                "item_id": item_ids,
                "event_timestamp": [datetime.now()] * len(item_ids)
            })
            
            # Retrieve features
            features = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_service
            ).to_df()
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to retrieve item features: {e}")
            return pd.DataFrame()
    
    def get_online_features(self, entity_rows: List[Dict[str, Any]], feature_refs: List[str]) -> pd.DataFrame:
        """
        Get online features for real-time serving
        """
        try:
            features = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows
            )
            return features.to_df()
            
        except Exception as e:
            logger.error(f"Failed to retrieve online features: {e}")
            return pd.DataFrame()
    
    def update_user_features(self, user_id: str, features: Dict[str, Any]):
        """
        Update features for a specific user (for real-time updates)
        """
        try:
            # Create push source for real-time updates
            push_source = PushSource(
                name="user_features_push",
                batch_source="user_features"
            )
            
            # Prepare feature data
            feature_data = {
                "user_id": [user_id],
                "event_timestamp": [datetime.now()],
                **{k: [v] for k, v in features.items()}
            }
            
            df = pd.DataFrame(feature_data)
            
            # Push to feature store
            self.store.push(push_source, df)
            
            logger.info(f"Updated features for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update user features: {e}")
    
    def update_item_features(self, item_id: str, features: Dict[str, Any]):
        """
        Update features for a specific item (for real-time updates)
        """
        try:
            # Create push source for real-time updates
            push_source = PushSource(
                name="item_features_push",
                batch_source="item_features"
            )
            
            # Prepare feature data
            feature_data = {
                "item_id": [item_id],
                "event_timestamp": [datetime.now()],
                **{k: [v] for k, v in features.items()}
            }
            
            df = pd.DataFrame(feature_data)
            
            # Push to feature store
            self.store.push(push_source, df)
            
            logger.info(f"Updated features for item {item_id}")
            
        except Exception as e:
            logger.error(f"Failed to update item features: {e}")
    
    def cleanup_old_features(self, days_to_keep: int = 90):
        """
        Clean up old features to save storage space
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # This would require custom implementation depending on storage backend
            logger.info(f"Cleaning up features older than {cutoff_date}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old features: {e}")


class FeatureEngineering:
    """
    Advanced feature engineering for recommendation systems
    """
    
    @staticmethod
    def create_user_behavior_features(interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced user behavior features
        """
        features = pd.DataFrame()
        
        # Time-based features
        interactions_df['hour'] = pd.to_datetime(interactions_df['timestamp']).dt.hour
        interactions_df['day_of_week'] = pd.to_datetime(interactions_df['timestamp']).dt.dayofweek
        interactions_df['month'] = pd.to_datetime(interactions_df['timestamp']).dt.month
        
        # User activity patterns
        user_activity = interactions_df.groupby('user_id').agg({
            'hour': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12,
            'day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
            'rating': ['mean', 'std', 'count'],
            'price': ['mean', 'min', 'max'],
            'category': lambda x: x.nunique()
        }).reset_index()
        
        user_activity.columns = [
            'user_id', 'preferred_hour', 'preferred_day', 'avg_rating', 
            'rating_std', 'total_interactions', 'avg_price', 'min_price', 
            'max_price', 'category_diversity'
        ]
        
        # Recency features
        current_time = pd.Timestamp.now()
        last_interactions = interactions_df.groupby('user_id')['timestamp'].max().reset_index()
        last_interactions['days_since_last_interaction'] = (
            current_time - pd.to_datetime(last_interactions['timestamp'])
        ).dt.days
        
        # Merge features
        features = user_activity.merge(last_interactions[['user_id', 'days_since_last_interaction']], 
                                     on='user_id')
        
        return features
    
    @staticmethod
    def create_item_trend_features(interactions_df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
        """
        Create time-based trend features for items
        """
        features = pd.DataFrame()
        
        # Convert timestamp
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        # Calculate trends for different time windows
        for window in [7, 30, 90]:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=window)
            
            recent_interactions = interactions_df[interactions_df['timestamp'] >= cutoff_date]
            
            item_trends = recent_interactions.groupby('item_id').agg({
                'rating': ['mean', 'count'],
                'user_id': 'nunique',
                'price': 'mean'
            }).reset_index()
            
            item_trends.columns = [
                f'item_id', f'avg_rating_{window}d', f'review_count_{window}d',
                f'unique_users_{window}d', f'avg_price_{window}d'
            ]
            
            if features.empty:
                features = item_trends
            else:
                features = features.merge(item_trends, on='item_id')
        
        return features
    
    @staticmethod
    def create_collaborative_features(interactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create collaborative filtering based features
        """
        features = pd.DataFrame()
        
        # User-item interaction matrix features
        user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        # Calculate similarity-based features
        from sklearn.metrics.pairwise import cosine_similarity
        
        # User similarity features
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity, 
            index=user_item_matrix.index, 
            columns=user_item_matrix.index
        )
        
        # Item similarity features
        item_similarity = cosine_similarity(user_item_matrix.T)
        item_similarity_df = pd.DataFrame(
            item_similarity, 
            index=user_item_matrix.columns, 
            columns=user_item_matrix.columns
        )
        
        return features
