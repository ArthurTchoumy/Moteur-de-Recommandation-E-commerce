"""
Collaborative Filtering Model using PySpark ALS
Implementation for e-commerce recommendation engine
"""

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringModel:
    """
    Collaborative Filtering using Alternating Least Squares (ALS)
    for user-item recommendations
    """
    
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def prepare_data(self, interactions_df):
        """
        Prepare interaction data for ALS training
        Args:
            interactions_df: DataFrame with user_id, item_id, rating, timestamp
        """
        logger.info("Preparing data for ALS model")
        
        # Create mappings for user and item IDs to continuous integers
        unique_users = interactions_df.select('user_id').distinct().collect()
        unique_items = interactions_df.select('item_id').distinct().collect()
        
        self.user_mapping = {row['user_id']: idx for idx, row in enumerate(unique_users)}
        self.item_mapping = {row['item_id']: idx for idx, row in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        # Convert to Spark DataFrame with mapped IDs
        pandas_df = interactions_df.toPandas()
        pandas_df['user_idx'] = pandas_df['user_id'].map(self.user_mapping)
        pandas_df['item_idx'] = pandas_df['item_id'].map(self.item_mapping)
        
        spark_df = self.spark.createDataFrame(pandas_df[['user_idx', 'item_idx', 'rating']])
        
        return spark_df
    
    def train(self, train_data, validation_data=None, **als_params):
        """
        Train the ALS model
        Args:
            train_data: Training DataFrame with user_idx, item_idx, rating
            validation_data: Optional validation data
            als_params: Parameters for ALS model
        """
        logger.info("Training ALS collaborative filtering model")
        
        # Default ALS parameters
        default_params = {
            'rank': 50,
            'maxIter': 10,
            'regParam': 0.1,
            'alpha': 1.0,
            'userCol': 'user_idx',
            'itemCol': 'item_idx',
            'ratingCol': 'rating',
            'coldStartStrategy': 'drop',
            'nonnegative': True
        }
        
        # Update with provided parameters
        params = {**default_params, **als_params}
        
        # Create ALS model
        als = ALS(**params)
        
        # Train model
        self.model = als.fit(train_data)
        
        # Evaluate on validation set if provided
        if validation_data is not None:
            predictions = self.model.transform(validation_data)
            evaluator = RegressionEvaluator(
                metricName="rmse", 
                labelCol="rating", 
                predictionCol="prediction"
            )
            rmse = evaluator.evaluate(predictions)
            logger.info(f"Validation RMSE: {rmse:.4f}")
            
            # Calculate additional metrics
            mae = RegressionEvaluator(
                metricName="mae", 
                labelCol="rating", 
                predictionCol="prediction"
            ).evaluate(predictions)
            
            logger.info(f"Validation MAE: {mae:.4f}")
            
            return {'rmse': rmse, 'mae': mae}
        
        return None
    
    def hyperparameter_tuning(self, train_data, validation_data):
        """
        Perform hyperparameter tuning using cross-validation
        """
        logger.info("Performing hyperparameter tuning")
        
        als = ALS(
            userCol='user_idx',
            itemCol='item_idx',
            ratingCol='rating',
            coldStartStrategy='drop'
        )
        
        # Parameter grid
        param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 20, 50]) \
            .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(als.maxIter, [5, 10, 20]) \
            .build()
        
        # Evaluator
        evaluator = RegressionEvaluator(
            metricName="rmse", 
            labelCol="rating", 
            predictionCol="prediction"
        )
        
        # Cross-validation
        crossval = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3
        )
        
        cv_model = crossval.fit(train_data)
        best_model = cv_model.bestModel
        
        logger.info(f"Best rank: {best_model.rank}")
        logger.info(f"Best regParam: {best_model._java_obj.getRegParam()}")
        logger.info(f"Best maxIter: {best_model._java_obj.getMaxIter()}")
        
        self.model = best_model
        return cv_model
    
    def recommend_for_users(self, user_ids: List[str], num_items: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for specific users
        Args:
            user_ids: List of original user IDs
            num_items: Number of recommendations per user
        Returns:
            Dictionary mapping user_id to list of (item_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Map user IDs to indices
        user_indices = [self.user_mapping.get(uid) for uid in user_ids]
        valid_users = [(uid, idx) for uid, idx in zip(user_ids, user_indices) if idx is not None]
        
        if not valid_users:
            logger.warning("No valid users found for recommendations")
            return {}
        
        user_indices_df = self.spark.createDataFrame(
            [(idx,) for _, idx in valid_users],
            ["user_idx"]
        )
        
        # Generate recommendations
        recommendations = self.model.recommendForUserSubset(user_indices_df, num_items)
        
        # Convert to dictionary format
        results = {}
        for row in recommendations.collect():
            original_user_id = self.reverse_user_mapping[row['user_idx']]
            recs = []
            for rec in row['recommendations']:
                original_item_id = self.reverse_item_mapping[rec['item_idx']]
                recs.append((original_item_id, float(rec['rating'])))
            results[original_user_id] = recs
        
        return results
    
    def recommend_for_items(self, item_ids: List[str], num_users: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find similar items based on user preferences
        Args:
            item_ids: List of original item IDs
            num_users: Number of similar users to consider
        Returns:
            Dictionary mapping item_id to list of (user_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Map item IDs to indices
        item_indices = [self.item_mapping.get(iid) for iid in item_ids]
        valid_items = [(iid, idx) for iid, idx in zip(item_ids, item_indices) if idx is not None]
        
        if not valid_items:
            logger.warning("No valid items found")
            return {}
        
        item_indices_df = self.spark.createDataFrame(
            [(idx,) for _, idx in valid_items],
            ["item_idx"]
        )
        
        # Generate similar users for items
        similar_users = self.model.recommendForItemSubset(item_indices_df, num_users)
        
        # Convert to dictionary format
        results = {}
        for row in similar_users.collect():
            original_item_id = self.reverse_item_mapping[row['item_idx']]
            similar = []
            for sim in row['recommendations']:
                original_user_id = self.reverse_user_mapping[sim['user_idx']]
                similar.append((original_user_id, float(sim['rating'])))
            results[original_item_id] = similar
        
        return results
    
    def get_user_factors(self, user_id: str) -> Optional[np.ndarray]:
        """Get user latent factors"""
        if self.model is None or user_id not in self.user_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        user_factors = self.model.userFactors.filter(f"id = {user_idx}").collect()
        
        if user_factors:
            return np.array(user_factors[0]['features'])
        return None
    
    def get_item_factors(self, item_id: str) -> Optional[np.ndarray]:
        """Get item latent factors"""
        if self.model is None or item_id not in self.item_mapping:
            return None
        
        item_idx = self.item_mapping[item_id]
        item_factors = self.model.itemFactors.filter(f"id = {item_idx}").collect()
        
        if item_factors:
            return np.array(item_factors[0]['features'])
        return None
    
    def calculate_similarity(self, id1: str, id2: str, is_user: bool = True) -> float:
        """
        Calculate cosine similarity between two users or items
        """
        if is_user:
            factors1 = self.get_user_factors(id1)
            factors2 = self.get_user_factors(id2)
        else:
            factors1 = self.get_item_factors(id1)
            factors2 = self.get_item_factors(id2)
        
        if factors1 is None or factors2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(factors1, factors2)
        norm1 = np.linalg.norm(factors1)
        norm2 = np.linalg.norm(factors2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = ALS.load(path)
        logger.info(f"Model loaded from {path}")


def create_spark_session(app_name: str = "EcommerceCF") -> SparkSession:
    """Create and configure Spark session"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    return spark
