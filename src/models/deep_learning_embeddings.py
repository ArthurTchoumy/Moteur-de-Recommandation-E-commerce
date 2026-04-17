"""
Deep Learning Model using TensorFlow Embeddings
Implementation for e-commerce recommendation engine
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses, callbacks
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLearningEmbeddingsModel:
    """
    Deep Learning model using embeddings for user-item recommendations
    Implements both Matrix Factorization and Neural Collaborative Filtering
    """
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 hidden_layers: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.model = None
        self.n_users = 0
        self.n_items = 0
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        Args:
            df: DataFrame with columns user_id, item_id, rating
        Returns:
            Tuple of (user_indices, item_indices, ratings)
        """
        logger.info("Preparing data for deep learning model")
        
        # Encode user and item IDs
        self.n_users = df['user_id'].nunique()
        self.n_items = df['item_id'].nunique()
        
        user_indices = self.user_encoder.fit_transform(df['user_id'])
        item_indices = self.item_encoder.fit_transform(df['item_id'])
        ratings = df['rating'].values
        
        logger.info(f"Encoded {self.n_users} users and {self.n_items} items")
        
        return user_indices, item_indices, ratings
    
    def build_matrix_factorization_model(self) -> Model:
        """
        Build Matrix Factorization model using embeddings
        """
        # User embedding
        user_input = layers.Input(shape=(), name='user_id')
        user_embedding = layers.Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding)
        
        # Item embedding
        item_input = layers.Input(shape=(), name='item_id')
        item_embedding = layers.Embedding(
            input_dim=self.n_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)
        item_vec = layers.Flatten(name='item_flatten')(item_embedding)
        
        # Dot product for prediction
        dot_product = layers.Dot(axes=1, name='dot_product')([user_vec, item_vec])
        
        # Add bias terms
        user_bias = layers.Embedding(input_dim=self.n_users, output_dim=1, name='user_bias')(user_input)
        user_bias = layers.Flatten(name='user_bias_flat')(user_bias)
        
        item_bias = layers.Embedding(input_dim=self.n_items, output_dim=1, name='item_bias')(item_input)
        item_bias = layers.Flatten(name='item_bias_flat')(item_bias)
        
        # Global bias
        global_bias = layers.Embedding(input_dim=1, output_dim=1, name='global_bias')(layers.Input(shape=()))
        global_bias = layers.Flatten(name='global_bias_flat')(global_bias)
        
        # Final prediction
        output = layers.Add(name='output')([
            dot_product,
            user_bias,
            item_bias,
            global_bias
        ])
        
        model = Model(
            inputs=[user_input, item_input, layers.Input(shape=())],
            outputs=output
        )
        
        return model
    
    def build_neural_cf_model(self) -> Model:
        """
        Build Neural Collaborative Filtering model
        """
        # User embedding
        user_input = layers.Input(shape=(), name='user_id')
        user_embedding = layers.Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding)
        
        # Item embedding
        item_input = layers.Input(shape=(), name='item_id')
        item_embedding = layers.Embedding(
            input_dim=self.n_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)
        item_vec = layers.Flatten(name='item_flatten')(item_embedding)
        
        # Concatenate embeddings
        concat = layers.Concatenate(name='concat')([user_vec, item_vec])
        
        # Hidden layers
        x = concat
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation='relu', name=f'hidden_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Output layer
        output = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=[user_input, item_input], outputs=output)
        
        return model
    
    def build_hybrid_model(self) -> Model:
        """
        Build hybrid model combining Matrix Factorization and Neural CF
        """
        # User and item inputs
        user_input = layers.Input(shape=(), name='user_id')
        item_input = layers.Input(shape=(), name='item_id')
        
        # User embedding
        user_embedding = layers.Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        user_vec = layers.Flatten(name='user_flatten')(user_embedding)
        
        # Item embedding
        item_embedding = layers.Embedding(
            input_dim=self.n_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)
        item_vec = layers.Flatten(name='item_flatten')(item_embedding)
        
        # Matrix Factorization path
        mf_dot = layers.Dot(axes=1, name='mf_dot')([user_vec, item_vec])
        
        # Neural CF path
        concat = layers.Concatenate(name='concat')([user_vec, item_vec])
        x = concat
        
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation='relu', name=f'hidden_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        neural_output = layers.Dense(1, activation='linear', name='neural_output')(x)
        
        # Combine both paths
        output = layers.Add(name='output')([mf_dot, neural_output])
        
        model = Model(inputs=[user_input, item_input], outputs=output)
        
        return model
    
    def train(self, 
              user_indices: np.ndarray,
              item_indices: np.ndarray,
              ratings: np.ndarray,
              model_type: str = 'hybrid',
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 256,
              early_stopping_patience: int = 10) -> Dict:
        """
        Train the deep learning model
        """
        logger.info(f"Training {model_type} model")
        
        # Build model
        if model_type == 'matrix_factorization':
            self.model = self.build_matrix_factorization_model()
            # Prepare dummy global bias input
            global_bias_input = np.zeros(len(user_indices))
            X_train = [user_indices, item_indices, global_bias_input]
        else:
            self.model = self.build_neural_cf_model() if model_type == 'neural_cf' else self.build_hybrid_model()
            X_train = [user_indices, item_indices]
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses.MeanSquaredError(),
            metrics=['mae']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, ratings,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info(f"Training completed. Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return history.history
    
    def predict(self, user_ids: List[str], item_ids: List[str]) -> np.ndarray:
        """
        Predict ratings for user-item pairs
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Encode IDs
        user_indices = self.user_encoder.transform(user_ids)
        item_indices = self.item_encoder.transform(item_ids)
        
        # Make predictions
        if 'global_bias' in self.model.input_names:
            global_bias_input = np.zeros(len(user_ids))
            predictions = self.model.predict([user_indices, item_indices, global_bias_input])
        else:
            predictions = self.model.predict([user_indices, item_indices])
        
        return predictions.flatten()
    
    def recommend_for_users(self, 
                           user_ids: List[str], 
                           item_candidates: List[str],
                           num_recommendations: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations for users
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        recommendations = {}
        
        for user_id in user_ids:
            # Create user-item pairs
            user_item_pairs = [(user_id, item_id) for item_id in item_candidates]
            users, items = zip(*user_item_pairs)
            
            # Predict ratings
            predictions = self.predict(list(users), list(items))
            
            # Sort by prediction score
            item_scores = list(zip(item_candidates, predictions))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            top_recommendations = item_scores[:num_recommendations]
            recommendations[user_id] = top_recommendations
        
        return recommendations
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get user embedding vector"""
        if self.model is None or user_id not in self.user_encoder.classes_:
            return None
        
        user_idx = self.user_encoder.transform([user_id])[0]
        embedding_layer = self.model.get_layer('user_embedding')
        embedding = embedding_layer(np.array([user_idx]))[0]
        
        return embedding.numpy()
    
    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get item embedding vector"""
        if self.model is None or item_id not in self.item_encoder.classes_:
            return None
        
        item_idx = self.item_encoder.transform([item_id])[0]
        embedding_layer = self.model.get_layer('item_embedding')
        embedding = embedding_layer(np.array([item_idx]))[0]
        
        return embedding.numpy()
    
    def find_similar_items(self, 
                          item_id: str, 
                          item_candidates: List[str],
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar items based on embedding similarity
        """
        target_embedding = self.get_item_embedding(item_id)
        if target_embedding is None:
            return []
        
        similarities = []
        for candidate_id in item_candidates:
            if candidate_id == item_id:
                continue
                
            candidate_embedding = self.get_item_embedding(candidate_id)
            if candidate_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(target_embedding, candidate_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(candidate_embedding)
            )
            similarities.append((candidate_id, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def calculate_user_similarity(self, user_id1: str, user_id2: str) -> float:
        """Calculate similarity between two users"""
        embedding1 = self.get_user_embedding(user_id1)
        embedding2 = self.get_user_embedding(user_id2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(path)
            
            # Save encoders
            import pickle
            with open(f"{path}_encoders.pkl", 'wb') as f:
                pickle.dump({
                    'user_encoder': self.user_encoder,
                    'item_encoder': self.item_encoder,
                    'n_users': self.n_users,
                    'n_items': self.n_items
                }, f)
            
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(path)
        
        # Load encoders
        import pickle
        with open(f"{path}_encoders.pkl", 'rb') as f:
            data = pickle.load(f)
            self.user_encoder = data['user_encoder']
            self.item_encoder = data['item_encoder']
            self.n_users = data['n_users']
            self.n_items = data['n_items']
        
        logger.info(f"Model loaded from {path}")


class ColdStartHandler:
    """
    Handle cold start problem for new users and items
    """
    
    def __init__(self):
        self.user_profiles = {}
        self.item_categories = {}
        
    def create_user_profile(self, user_id: str, preferences: Dict):
        """Create profile for new user based on preferences"""
        self.user_profiles[user_id] = {
            'categories': preferences.get('categories', []),
            'price_range': preferences.get('price_range', [0, float('inf')]),
            'brands': preferences.get('brands', []),
            'created_at': pd.Timestamp.now()
        }
    
    def get_cold_start_recommendations(self, 
                                     user_id: str, 
                                     item_catalog: pd.DataFrame,
                                     num_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Get recommendations for new user based on profile"""
        if user_id not in self.user_profiles:
            # Return popular items if no profile
            return self._get_popular_items(item_catalog, num_recommendations)
        
        profile = self.user_profiles[user_id]
        
        # Filter items based on profile
        filtered_items = item_catalog.copy()
        
        # Filter by categories
        if profile['categories']:
            filtered_items = filtered_items[
                filtered_items['category'].isin(profile['categories'])
            ]
        
        # Filter by price range
        min_price, max_price = profile['price_range']
        filtered_items = filtered_items[
            (filtered_items['price'] >= min_price) & 
            (filtered_items['price'] <= max_price)
        ]
        
        # Filter by brands
        if profile['brands']:
            filtered_items = filtered_items[
                filtered_items['brand'].isin(profile['brands'])
            ]
        
        # Sort by rating and popularity
        filtered_items = filtered_items.sort_values(
            ['rating', 'review_count'], 
            ascending=[False, False]
        )
        
        recommendations = []
        for _, item in filtered_items.head(num_recommendations).iterrows():
            score = self._calculate_cold_start_score(item, profile)
            recommendations.append((item['item_id'], score))
        
        return recommendations
    
    def _get_popular_items(self, item_catalog: pd.DataFrame, num_items: int) -> List[Tuple[str, float]]:
        """Get popular items as fallback"""
        popular_items = item_catalog.sort_values(
            ['rating', 'review_count'], 
            ascending=[False, False]
        ).head(num_items)
        
        return [(item['item_id'], item['rating']) for _, item in popular_items.iterrows()]
    
    def _calculate_cold_start_score(self, item: pd.Series, profile: Dict) -> float:
        """Calculate score for cold start recommendation"""
        score = item.get('rating', 0) * 0.5
        
        # Boost for category match
        if item.get('category') in profile['categories']:
            score += 0.3
        
        # Boost for brand match
        if item.get('brand') in profile['brands']:
            score += 0.2
        
        return min(score, 5.0)  # Cap at 5.0
