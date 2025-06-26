# Magnus Chess AI ♟️

Advanced chess move prediction AI inspired by Magnus Carlsen's playing style, built with modern MLOps practices.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd What-Would---DO

# Install dependencies
pip install -r requirements.txt

# Setup DVC (data version control)
dvc pull

# Train a fast model
python src/training/train_fast_magnus.py

# Run inference
python src/inference/inference_class.py

# Launch model dashboard
streamlit run src/mlops/model_dashboard.py
```

## 📁 Project Structure

```
What-Would---DO/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── checkpoints/         # Saved model weights (DVC tracked)
│   │   └── architectures/       # Model definitions
│   ├── training/                 # Training scripts
│   │   ├── train_fast_magnus.py # Quick training script
│   │   └── train_enhanced_magnus.py # Full training pipeline
│   ├── inference/                # Model inference
│   │   └── inference_class.py   # Inference utilities
│   ├── mlops/                    # MLOps and model management
│   │   ├── mlops_enhanced_manager.py # Advanced model versioning
│   │   └── model_dashboard.py   # Interactive dashboard
│   ├── data/                     # Data processing
│   │   ├── carlsen-games.pgn    # Magnus Carlsen games (DVC tracked)
│   │   └── prepare_data.py      # Data preprocessing
│   └── utils/                    # Shared utilities
├── tests/                        # Test files
├── docs/                         # Documentation
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── Frontend/                     # React web interface
├── Backend/                      # FastAPI backend
├── requirements.txt              # Python dependencies
├── dvc.yaml                      # DVC pipeline definition
└── .dvc/                         # DVC configuration
```

## 🧠 Features

### Chess AI
- **Magnus-inspired model**: Trained on Magnus Carlsen's games
- **Multiple architectures**: Fast and enhanced training modes
- **Real-time inference**: Quick move prediction
- **Position evaluation**: Advanced position analysis

### MLOps Pipeline
- **Experiment Tracking**: MLflow integration
- **Data Versioning**: DVC for datasets and models
- **Model Registry**: Comprehensive versioning and metadata
- **Automated Pipelines**: DVC pipelines for reproducible training
- **Interactive Dashboard**: Streamlit-based model management

### Model Management
- **Version Control**: Git + DVC for complete reproducibility
- **Model Lineage**: Track model ancestry and evolution
- **Performance Monitoring**: Automated metrics tracking
- **Deployment Ready**: Easy model serving and API integration

## 🛠️ Development

### Training Models

```bash
# Fast training (30 epochs, quick iteration)
python src/training/train_fast_magnus.py

# Enhanced training (100 epochs, full pipeline)
python src/training/train_enhanced_magnus.py

# Custom configuration
python src/training/train_fast_magnus.py --config configs/custom_config.yaml
```

### DVC Pipeline

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train_fast_model

# Check pipeline status
dvc dag
```

### Model Management

```bash
# Launch dashboard
streamlit run src/mlops/model_dashboard.py

# List model versions
python -c "from src.mlops.mlops_enhanced_manager import EnhancedMagnusModelManager; print(EnhancedMagnusModelManager().list_versions())"
```

## 📊 Model Performance

Recent training results:
- **Fast Model**: 8.6% top-1 accuracy (1.58 min training)
- **Enhanced Model**: Higher accuracy with comprehensive logging
- **Model Size**: ~1.38M parameters (optimized for inference speed)

## 🔧 Configuration

Key configuration files:
- `configs/training_config.yaml` - Training hyperparameters
- `dvc.yaml` - Data pipeline definition
- `pyproject.toml` - Project metadata and dependencies

## 📈 Monitoring

The project includes comprehensive monitoring:
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data and model versioning
- **Dashboard**: Interactive model comparison and management
- **Git**: Source code versioning and collaboration

## 🚀 Deployment

Models are deployment-ready with:
- **FastAPI backend**: REST API for model serving
- **React frontend**: Interactive chess interface
- **Docker support**: Containerized deployment
- **Model versioning**: Easy rollback and A/B testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Train and validate your model changes
4. Update documentation
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 Roadmap

- [ ] Multi-GPU training support
- [ ] Advanced position embeddings
- [ ] Opening book integration
- [ ] Tournament-style evaluation
- [ ] Real-time game analysis
- [ ] Mobile app interface

---

Built with ❤️ using PyTorch, MLflow, DVC, and modern MLOps practices.
