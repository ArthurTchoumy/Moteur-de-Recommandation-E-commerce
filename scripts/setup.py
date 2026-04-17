"""
Setup script for E-commerce Recommendation Engine
Automates installation and configuration of all components
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationEngineSetup:
    """Setup and configuration manager for the recommendation engine"""
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.config_dir = self.project_root / "configs"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error("requirements.txt not found!")
            return False
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            logger.info("Dependencies installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup_redis(self):
        """Setup Redis configuration"""
        logger.info("Setting up Redis configuration...")
        
        redis_config = {
            "bind": "127.0.0.1",
            "port": 6379,
            "timeout": 0,
            "tcp-keepalive": 300,
            "daemonize": "no",
            "supervised": "no",
            "pidfile": "/var/run/redis_6379.pid",
            "loglevel": "notice",
            "logfile": "",
            "databases": 16,
            "save": ["900 1", "300 10", "60 10000"],
            "stop-writes-on-bgsave-error": "yes",
            "rdbcompression": "yes",
            "rdbchecksum": "yes",
            "dbfilename": "dump.rdb",
            "dir": "./",
            "slaveof": "no",
            "masterauth": "",
            "slave-serve-stale-data": "yes",
            "slave-read-only": "yes",
            "repl-diskless-sync": "no",
            "repl-diskless-sync-delay": 5,
            "slave-priority": 100,
            "maxmemory-policy": "allkeys-lru",
            "appendonly": "yes",
            "appendfilename": "appendonly.aof",
            "appendfsync": "everysec",
            "no-appendfsync-on-rewrite": "no",
            "auto-aof-rewrite-percentage": 100,
            "auto-aof-rewrite-min-size": "64mb",
            "aof-load-truncated": "yes",
            "lua-time-limit": 5000,
            "slowlog-log-slower-than": 10000,
            "slowlog-max-len": 128,
            "notify-keyspace-events": "",
            "hash-max-ziplist-entries": 512,
            "hash-max-ziplist-value": 64,
            "list-max-ziplist-size": -2,
            "list-compress-depth": 0,
            "set-max-intset-entries": 512,
            "zset-max-ziplist-entries": 128,
            "zset-max-ziplist-value": 64,
            "hll-sparse-max-bytes": 3000,
            "activerehashing": "yes",
            "client-output-buffer-limit": ["normal 0 0 0", "slave 256mb 64mb 60", "pubsub 32mb 8mb 60"],
            "hz": 10,
            "aof-rewrite-incremental-fsync": "yes"
        }
        
        config_file = self.config_dir / "redis.conf"
        
        with open(config_file, 'w') as f:
            for key, value in redis_config.items():
                if isinstance(value, list):
                    for v in value:
                        f.write(f"{key} {v}\n")
                else:
                    f.write(f"{key} {value}\n")
        
        logger.info(f"Redis configuration saved to {config_file}")
        return True
    
    def setup_spark(self):
        """Setup Spark configuration"""
        logger.info("Setting up Spark configuration...")
        
        spark_config = {
            "spark.app.name": "EcommerceRecommendationEngine",
            "spark.master": "local[*]",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
            "spark.driver.memory": "4g",
            "spark.driver.maxResultSize": "2g",
            "spark.executor.memory": "4g",
            "spark.executor.cores": "2",
            "spark.default.parallelism": "8",
            "spark.sql.shuffle.partitions": "200"
        }
        
        config_file = self.config_dir / "spark.conf"
        
        with open(config_file, 'w') as f:
            for key, value in spark_config.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Spark configuration saved to {config_file}")
        return True
    
    def setup_feast(self):
        """Setup Feast feature store configuration"""
        logger.info("Setting up Feast feature store...")
        
        feast_config = f"""
project: ecommerce_features
registry: {self.config_dir / "feature_repo" / "registry.db"}
provider: local
offline_store:
    type: file
online_store:
    type: redis
    connection_string: "localhost:6379"
"""
        
        feast_repo_dir = self.config_dir / "feature_repo"
        feast_repo_dir.mkdir(exist_ok=True)
        
        config_file = feast_repo_dir / "feature_store.yaml"
        
        with open(config_file, 'w') as f:
            f.write(feast_config)
        
        logger.info(f"Feast configuration saved to {config_file}")
        return True
    
    def create_sample_data(self):
        """Create sample data for testing"""
        logger.info("Creating sample data...")
        
        # Sample users data
        np.random.seed(42)
        n_users = 1000
        
        users_data = {
            'user_id': [f'user_{i}' for i in range(n_users)],
            'email': [f'user{i}@example.com' for i in range(n_users)],
            'join_date': pd.date_range('2020-01-01', periods=n_users, freq='H'),
            'age': np.random.randint(18, 65, n_users),
            'gender': np.random.choice(['M', 'F', 'Other'], n_users),
            'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_users)
        }
        
        users_df = pd.DataFrame(users_data)
        users_df.to_parquet(self.data_dir / "users.parquet", index=False)
        
        # Sample items data
        n_items = 5000
        
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty']
        brands = ['Apple', 'Samsung', 'Nike', 'Adidas', 'Sony', 'LG', 'Dell', 'HP']
        
        items_data = {
            'item_id': [f'item_{i}' for i in range(n_items)],
            'title': [f'Product {i}' for i in range(n_items)],
            'category': np.random.choice(categories, n_items),
            'brand': np.random.choice(brands, n_items),
            'price': np.random.uniform(10, 1000, n_items),
            'description': [f'Description for product {i}' for i in range(n_items)],
            'availability': np.random.choice(['in_stock', 'out_of_stock'], n_items, p=[0.9, 0.1]),
            'discount_percentage': np.random.uniform(0, 50, n_items)
        }
        
        items_df = pd.DataFrame(items_data)
        items_df.to_parquet(self.data_dir / "items.parquet", index=False)
        
        # Sample interactions data
        n_interactions = 50000
        
        user_ids = np.random.choice(users_df['user_id'], n_interactions)
        item_ids = np.random.choice(items_df['item_id'], n_interactions)
        
        interactions_data = {
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': np.random.uniform(1, 5, n_interactions),
            'timestamp': pd.date_range('2023-01-01', periods=n_interactions, freq='min'),
            'review_text': [f'Review text {i}' for i in range(n_interactions)],
            'helpful_votes': np.random.randint(0, 100, n_interactions)
        }
        
        # Add price from items
        item_prices = items_df.set_index('item_id')['price'].to_dict()
        interactions_data['price'] = [item_prices[item_id] for item_id in item_ids]
        
        # Add category from items
        item_categories = items_df.set_index('item_id')['category'].to_dict()
        interactions_data['category'] = [item_categories[item_id] for item_id in item_ids]
        
        interactions_df = pd.DataFrame(interactions_data)
        interactions_df.to_parquet(self.data_dir / "interactions.parquet", index=False)
        
        logger.info("Sample data created successfully!")
        return True
    
    def create_environment_config(self):
        """Create environment configuration file"""
        logger.info("Creating environment configuration...")
        
        env_config = {
            "development": {
                "api_host": "localhost",
                "api_port": 8000,
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_db": 0,
                "spark_master": "local[*]",
                "feast_repo_path": str(self.config_dir / "feature_repo"),
                "log_level": "INFO",
                "cache_ttl_seconds": 3600,
                "max_recommendations": 50,
                "default_model": "hybrid"
            },
            "production": {
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "redis_host": "redis",
                "redis_port": 6379,
                "redis_db": 0,
                "spark_master": "spark://spark-master:7077",
                "feast_repo_path": "/app/configs/feature_repo",
                "log_level": "WARNING",
                "cache_ttl_seconds": 1800,
                "max_recommendations": 20,
                "default_model": "hybrid"
            }
        }
        
        config_file = self.config_dir / "environment.json"
        
        with open(config_file, 'w') as f:
            json.dump(env_config, f, indent=2)
        
        logger.info(f"Environment configuration saved to {config_file}")
        return True
    
    def create_startup_scripts(self):
        """Create startup scripts for different components"""
        logger.info("Creating startup scripts...")
        
        # API startup script
        api_script = f"""@echo off
echo Starting Recommendation API...
cd /d "{self.project_root}"
set PYTHONPATH={self.project_root}
python -m uvicorn src.serving.recommendation_api:app --host 0.0.0.0 --port 8000 --reload
pause
"""
        
        api_script_file = self.project_root / "start_api.bat"
        with open(api_script_file, 'w') as f:
            f.write(api_script)
        
        # Streamlit user interface startup script
        ui_script = f"""@echo off
echo Starting User Interface...
cd /d "{self.project_root}"
set PYTHONPATH={self.project_root}
streamlit run src/ui/streamlit_app.py --server.port 8501
pause
"""
        
        ui_script_file = self.project_root / "start_ui.bat"
        with open(ui_script_file, 'w') as f:
            f.write(ui_script)
        
        # Admin dashboard startup script
        admin_script = f"""@echo off
echo Starting Admin Dashboard...
cd /d "{self.project_root}"
set PYTHONPATH={self.project_root}
streamlit run src/ui/admin_dashboard.py --server.port 8502
pause
"""
        
        admin_script_file = self.project_root / "start_admin.bat"
        with open(admin_script_file, 'w') as f:
            f.write(admin_script)
        
        logger.info("Startup scripts created successfully!")
        return True
    
    def create_readme(self):
        """Create README file with setup instructions"""
        logger.info("Creating README file...")
        
        readme_content = """# E-commerce Recommendation Engine

A comprehensive recommendation engine built with modern ML technologies for e-commerce platforms.

## 🚀 Features

- **Collaborative Filtering**: PySpark ALS for scalable user-item recommendations
- **Deep Learning**: TensorFlow embeddings for neural collaborative filtering
- **Feature Store**: Feast for real-time feature serving
- **High Performance**: Redis caching for <50ms response times
- **A/B Testing**: Statistical testing framework for model comparison
- **Cold Start**: Intelligent recommendations for new users
- **User Interface**: Streamlit-based UI for users and administrators

## 📋 Requirements

- Python 3.8+
- Java 8+ (for Spark)
- Redis Server
- 8GB+ RAM recommended

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd projet3.0
python scripts/setup.py
```

### 2. Start Redis Server
```bash
redis-server configs/redis.conf
```

### 3. Start Services

#### Option 1: Use startup scripts
```bash
# Start API server
start_api.bat

# Start user interface
start_ui.bat

# Start admin dashboard
start_admin.bat
```

#### Option 2: Manual start
```bash
# API Server
python -m uvicorn src.serving.recommendation_api:app --host 0.0.0.0 --port 8000 --reload

# User Interface
streamlit run src/ui/streamlit_app.py --server.port 8501

# Admin Dashboard
streamlit run src/ui/admin_dashboard.py --server.port 8502
```

## 🌐 Access Points

- **API Documentation**: http://localhost:8000/docs
- **User Interface**: http://localhost:8501
- **Admin Dashboard**: http://localhost:8502

## 📊 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │  Admin Dashboard│    │   API Gateway   │
│   (Streamlit)   │    │   (Streamlit)   │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
              ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
              │  Collaborative  │              │  Deep Learning  │              │  Feature Store  │
              │   Filtering     │              │   Embeddings    │              │    (Feast)      │
              │   (PySpark)     │              │  (TensorFlow)   │              │                 │
              └─────────────────┘              └─────────────────┘              └─────────────────┘
                       │                                 │                                 │
                       └─────────────────────────────────┼─────────────────────────────────┘
                                                        │
                                              ┌─────────────────┐
                                              │     Redis       │
                                              │     Cache       │
                                              └─────────────────┘
```

## 🧪 A/B Testing

The system includes a comprehensive A/B testing framework:

- Statistical significance testing
- Multiple metrics support (CTR, conversion rate, revenue)
- Real-time monitoring
- Automated winner detection

## 📈 Performance Targets

- **API Latency**: <50ms (95th percentile)
- **Cache Hit Rate**: >80%
- **CTR Improvement**: +15% vs random recommendations
- **Scalability**: 10M users, 1M products

## 🔧 Configuration

Configuration files are located in the `configs/` directory:

- `redis.conf`: Redis server configuration
- `spark.conf`: Spark cluster settings
- `environment.json`: Environment-specific settings
- `feature_repo/`: Feast feature store configuration

## 📁 Project Structure

```
projet3.0/
├── src/
│   ├── models/              # ML models (Collaborative Filtering, Deep Learning)
│   ├── features/            # Feature engineering and Feast integration
│   ├── serving/             # FastAPI recommendation service
│   ├── cache/               # Redis caching layer
│   ├── ab_testing/          # A/B testing framework
│   ├── ui/                  # Streamlit interfaces
│   └── utils/               # Utility functions
├── configs/                 # Configuration files
├── data/                    # Data files (processed and raw)
├── notebooks/               # Jupyter notebooks for experimentation
├── tests/                   # Unit and integration tests
├── scripts/                 # Setup and utility scripts
└── logs/                    # Application logs
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📚 API Documentation

Once the API server is running, visit http://localhost:8000/docs for interactive API documentation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the logs in the `logs/` directory
2. Verify all services are running
3. Check API health endpoint: http://localhost:8000/health
4. Review configuration files in `configs/`
"""
        
        readme_file = self.project_root / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"README file created at {readme_file}")
        return True
    
    def run_setup(self):
        """Run complete setup process"""
        logger.info("Starting complete setup process...")
        
        setup_steps = [
            ("Installing dependencies", self.install_dependencies),
            ("Setting up Redis configuration", self.setup_redis),
            ("Setting up Spark configuration", self.setup_spark),
            ("Setting up Feast feature store", self.setup_feast),
            ("Creating sample data", self.create_sample_data),
            ("Creating environment configuration", self.create_environment_config),
            ("Creating startup scripts", self.create_startup_scripts),
            ("Creating README file", self.create_readme)
        ]
        
        failed_steps = []
        
        for step_name, step_func in setup_steps:
            logger.info(f"Running: {step_name}")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    logger.error(f"Failed: {step_name}")
                else:
                    logger.info(f"Completed: {step_name}")
            except Exception as e:
                failed_steps.append(step_name)
                logger.error(f"Error in {step_name}: {e}")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("SETUP SUMMARY")
        logger.info("="*50)
        
        if failed_steps:
            logger.error(f"Failed steps: {', '.join(failed_steps)}")
            logger.error("Setup completed with errors!")
            return False
        else:
            logger.info("All setup steps completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Start Redis server: redis-server configs/redis.conf")
            logger.info("2. Start API server: start_api.bat")
            logger.info("3. Start user interface: start_ui.bat")
            logger.info("4. Start admin dashboard: start_admin.bat")
            logger.info("\nAccess points:")
            logger.info("- API: http://localhost:8000")
            logger.info("- User Interface: http://localhost:8501")
            logger.info("- Admin Dashboard: http://localhost:8502")
            return True


def main():
    """Main setup function"""
    print("🚀 E-commerce Recommendation Engine Setup")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create setup instance
    setup = RecommendationEngineSetup(str(project_root))
    
    # Run setup
    success = setup.run_setup()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("📖 See README.md for next steps.")
    else:
        print("\n❌ Setup completed with errors!")
        print("🔧 Check the logs above for details.")
    
    return success


if __name__ == "__main__":
    main()
