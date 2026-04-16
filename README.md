# E-commerce Recommendation Engine

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
