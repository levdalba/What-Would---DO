# Magnus Chess AI - What Would Magnus Do?

A sophisticated chess AI system that learns from Magnus Carlsen's playing style using deep learning and MLOps best practices.

## 🚀 Features

- **Deep Learning Models**: Advanced neural networks trained on Magnus Carlsen's games
- **MLOps Pipeline**: Complete model versioning, tracking, and deployment
- **Data Version Control**: DVC for managing large datasets and models
- **Interactive Dashboard**: Model management and performance monitoring
- **Multiple Training Modes**: Fast prototyping and comprehensive training
- **Chess Engine Integration**: Compatible with standard chess engines

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model definitions
│   ├── training/                 # Training scripts
│   ├── inference/                # Inference and prediction
│   ├── mlops/                    # MLOps and model management
│   ├── data/                     # Data processing utilities
│   └── utils/                    # Shared utilities
├── models/                       # Saved models (DVC tracked)
├── tests/                        # Test files
├── docs/                         # Documentation
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── Backend/                      # API backend
├── Frontend/                     # React web interface
└── archive/                      # Archived/legacy files
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd What-Would---DO
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize DVC** (if not already done):
   ```bash
   dvc pull  # Download models and data
   ```

## 🚀 Quick Start

### Training a Model

**Fast Training** (for experimentation):
```bash
python src/training/train_fast_magnus.py
```

**Enhanced Training** (full pipeline):
```bash
python src/training/train_enhanced_magnus.py
```

### Model Inference

```bash
python src/inference/inference_class.py
```

### Model Management Dashboard

```bash
python src/mlops/dashboard.py
```

### MLflow UI

```bash
mlflow ui
```

## 📊 MLOps Features

- **Model Versioning**: Automatic versioning with Git commits and DVC
- **Experiment Tracking**: MLflow for metrics, parameters, and artifacts
- **Model Registry**: Centralized model management and promotion
- **Data Versioning**: DVC for large files and datasets
- **Automated Backups**: Local and cloud backup strategies
- **Performance Monitoring**: Model drift detection and performance tracking

## 🎯 Model Training

The system supports multiple training approaches:

1. **Fast Training**: Quick prototyping with reduced datasets
2. **Enhanced Training**: Full training with comprehensive logging
3. **Transfer Learning**: Fine-tuning from pre-trained models
4. **Hyperparameter Tuning**: Automated optimization

## 📈 Model Performance

Current best models achieve:
- **Move Prediction Accuracy**: ~85% top-1, ~95% top-3
- **Training Time**: 2-30 minutes depending on configuration
- **Model Size**: 1-10M parameters

## 🔧 Configuration

Models and training can be configured via:
- `configs/training_config.yaml`
- Command line arguments
- Environment variables

## 📚 Data

The system uses:
- **Magnus Carlsen Games**: PGN files with ~3000+ games
- **Processed Positions**: Extracted chess positions and moves
- **Augmented Data**: Generated variations and positions

## 🌐 Web Interface

Access the web interface at `http://localhost:3000` (Frontend) with:
- Interactive chess board
- Move prediction
- Game analysis
- Model comparison

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 📖 Documentation

- [Training Guide](docs/training.md)
- [MLOps Guide](docs/mlops.md)
- [API Reference](docs/api.md)
- [Architecture Overview](ARCHITECTURE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Acknowledgments

- Magnus Carlsen for the inspiration and games
- Chess.com and Lichess for game data
- Open source chess engines and libraries

## 📧 Contact

For questions or support, please open an issue or contact the team.

---

**What Would Magnus Do?** - *Bringing the world champion's intuition to your chess games.*
