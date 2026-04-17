"""
Redis Cache implementation for recommendation engine
Provides high-performance caching for recommendations and features
"""

import redis
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for Redis cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30


class RecommendationCache:
    """
    Redis-based cache for recommendation system
    Optimized for <50ms latency requirement
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=self.config.decode_responses,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _generate_key(self, prefix: str, identifier: str, params: Dict = None) -> str:
        """Generate cache key with parameters"""
        if params:
            # Sort parameters to ensure consistent key generation
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{prefix}:{identifier}:{param_hash}"
        return f"{prefix}:{identifier}"
    
    def cache_recommendations(self, 
                            user_id: str, 
                            recommendations: List[Dict[str, Any]], 
                            model_name: str = "default",
                            ttl_seconds: int = 3600) -> bool:
        """
        Cache recommendations for a user
        Args:
            user_id: User identifier
            recommendations: List of recommendation dicts with item_id, score, etc.
            model_name: Name of the model that generated recommendations
            ttl_seconds: Time to live in seconds
        """
        try:
            key = self._generate_key("recs", user_id, {"model": model_name})
            
            # Prepare cache data
            cache_data = {
                "user_id": user_id,
                "model_name": model_name,
                "recommendations": recommendations,
                "cached_at": datetime.now().isoformat(),
                "ttl": ttl_seconds
            }
            
            # Store in Redis
            self.redis_client.setex(
                key, 
                ttl_seconds, 
                json.dumps(cache_data, default=str)
            )
            
            logger.debug(f"Cached recommendations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache recommendations: {e}")
            return False
    
    def get_cached_recommendations(self, 
                                 user_id: str, 
                                 model_name: str = "default") -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached recommendations for a user
        """
        try:
            key = self._generate_key("recs", user_id, {"model": model_name})
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                
                # Check if cache is still valid
                cached_at = datetime.fromisoformat(data["cached_at"])
                if datetime.now() - cached_at < timedelta(seconds=data["ttl"]):
                    logger.debug(f"Retrieved cached recommendations for user {user_id}")
                    return data["recommendations"]
                else:
                    # Remove expired cache
                    self.redis_client.delete(key)
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached recommendations: {e}")
            return None
    
    def cache_user_features(self, 
                           user_id: str, 
                           features: Dict[str, Any], 
                           ttl_seconds: int = 1800) -> bool:
        """
        Cache user features
        """
        try:
            key = self._generate_key("features", "user", {"user_id": user_id})
            
            cache_data = {
                "user_id": user_id,
                "features": features,
                "cached_at": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                key, 
                ttl_seconds, 
                json.dumps(cache_data, default=str)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache user features: {e}")
            return False
    
    def get_cached_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached user features
        """
        try:
            key = self._generate_key("features", "user", {"user_id": user_id})
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data["features"]
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached user features: {e}")
            return None
    
    def cache_item_features(self, 
                           item_id: str, 
                           features: Dict[str, Any], 
                           ttl_seconds: int = 1800) -> bool:
        """
        Cache item features
        """
        try:
            key = self._generate_key("features", "item", {"item_id": item_id})
            
            cache_data = {
                "item_id": item_id,
                "features": features,
                "cached_at": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                key, 
                ttl_seconds, 
                json.dumps(cache_data, default=str)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache item features: {e}")
            return False
    
    def get_cached_item_features(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached item features
        """
        try:
            key = self._generate_key("features", "item", {"item_id": item_id})
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data["features"]
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached item features: {e}")
            return None
    
    def cache_model_predictions(self, 
                              model_name: str, 
                              predictions: Dict[str, Any], 
                              ttl_seconds: int = 7200) -> bool:
        """
        Cache model predictions for batch processing
        """
        try:
            key = self._generate_key("predictions", model_name)
            
            cache_data = {
                "model_name": model_name,
                "predictions": predictions,
                "cached_at": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                key, 
                ttl_seconds, 
                json.dumps(cache_data, default=str)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache model predictions: {e}")
            return False
    
    def get_cached_model_predictions(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached model predictions
        """
        try:
            key = self._generate_key("predictions", model_name)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data["predictions"]
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached model predictions: {e}")
            return None
    
    def cache_ab_test_results(self, 
                            test_id: str, 
                            results: Dict[str, Any], 
                            ttl_seconds: int = 86400) -> bool:
        """
        Cache A/B test results
        """
        try:
            key = self._generate_key("ab_test", test_id)
            
            cache_data = {
                "test_id": test_id,
                "results": results,
                "cached_at": datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                key, 
                ttl_seconds, 
                json.dumps(cache_data, default=str)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache A/B test results: {e}")
            return False
    
    def get_cached_ab_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached A/B test results
        """
        try:
            key = self._generate_key("ab_test", test_id)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data["results"]
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached A/B test results: {e}")
            return None
    
    def invalidate_user_cache(self, user_id: str) -> bool:
        """
        Invalidate all cache entries for a specific user
        """
        try:
            # Find all keys for this user
            pattern = f"*:{user_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate user cache: {e}")
            return False
    
    def invalidate_model_cache(self, model_name: str) -> bool:
        """
        Invalidate all cache entries for a specific model
        """
        try:
            # Find all keys for this model
            pattern = f"*:*{model_name}*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for model {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate model cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics
        """
        try:
            info = self.redis_client.info()
            
            stats = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total_requests = hits + misses
            
            if total_requests > 0:
                stats["hit_rate"] = hits / total_requests
            else:
                stats["hit_rate"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def cleanup_expired_cache(self) -> int:
        """
        Clean up expired cache entries (handled automatically by Redis TTL)
        This method can be used for manual cleanup if needed
        """
        try:
            # Redis automatically removes expired keys
            # This is a placeholder for any additional cleanup logic
            logger.info("Cache cleanup completed (Redis handles TTL automatically)")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return -1
    
    def health_check(self) -> bool:
        """
        Check if Redis cache is healthy
        """
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")


class CacheWarmer:
    """
    Cache warming utility to prepopulate cache with frequently accessed data
    """
    
    def __init__(self, cache: RecommendationCache):
        self.cache = cache
    
    def warm_popular_items(self, item_ids: List[str], item_features: pd.DataFrame):
        """
        Warm cache with popular item features
        """
        logger.info(f"Warming cache with {len(item_ids)} popular items")
        
        for item_id in item_ids:
            item_data = item_features[item_features['item_id'] == item_id]
            if not item_data.empty:
                features = item_data.iloc[0].to_dict()
                self.cache.cache_item_features(item_id, features)
    
    def warm_active_users(self, user_ids: List[str], user_features: pd.DataFrame):
        """
        Warm cache with active user features
        """
        logger.info(f"Warming cache with {len(user_ids)} active users")
        
        for user_id in user_ids:
            user_data = user_features[user_features['user_id'] == user_id]
            if not user_data.empty:
                features = user_data.iloc[0].to_dict()
                self.cache.cache_user_features(user_id, features)
    
    def warm_recommendations(self, 
                           user_ids: List[str], 
                           recommendations: Dict[str, List[Dict[str, Any]]],
                           model_name: str = "default"):
        """
        Warm cache with precomputed recommendations
        """
        logger.info(f"Warming cache with recommendations for {len(user_ids)} users")
        
        for user_id in user_ids:
            if user_id in recommendations:
                self.cache.cache_recommendations(
                    user_id, 
                    recommendations[user_id], 
                    model_name
                )


# Utility functions for cache management
def create_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Utility function to create consistent cache keys
    """
    key_parts = [prefix] + [str(arg) for arg in args]
    
    if kwargs:
        # Sort kwargs to ensure consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
    
    return ":".join(key_parts)


def serialize_object(obj: Any) -> str:
    """
    Serialize object for Redis storage
    """
    try:
        return pickle.dumps(obj)
    except Exception:
        return json.dumps(obj, default=str)


def deserialize_object(data: str, use_pickle: bool = False) -> Any:
    """
    Deserialize object from Redis storage
    """
    try:
        if use_pickle:
            return pickle.loads(data)
        else:
            return json.loads(data)
    except Exception:
        return data
