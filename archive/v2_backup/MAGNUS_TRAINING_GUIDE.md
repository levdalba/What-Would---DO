# Magnus Carlsen Style Chess Training Guide

## Overview

This guide explains how to train a neural network to mimic Magnus Carlsen's playing style using supervised learning. Unlike LC0 which requires massive self-play training, this approach directly learns from Magnus's actual games.

## Why This Approach Instead of LC0 Fine-tuning?

**LC0 cannot be fine-tuned** in the traditional sense because:

1. It's trained via self-play reinforcement learning, not supervised learning
2. Training requires weeks/months even on powerful hardware (hundreds of GPUs)
3. The training process is complex and requires specialized infrastructure

**Our supervised learning approach:**

-   Learns directly from Magnus's moves in actual games
-   Much faster training (hours instead of weeks)
-   Requires only standard GPU hardware
-   Can be trained on Colab with A100 GPU
-   Produces interpretable results

## Training Architecture

### 1. Data Processing Pipeline

-   **Input**: Magnus Carlsen PGN games (6k+ games)
-   **Features**: Board position + game context + position evaluation
-   **Target**: Magnus's actual move choice
-   **Outcome**: Game result for auxiliary learning

### 2. Model Architecture

-   **Board Encoder**: CNN to process 8x8x12 board representation
-   **Feature Encoder**: MLP for game context (material, castling, etc.)
-   **Move Predictor**: Combined network predicting move probabilities
-   **Outcome Predictor**: Auxiliary task for better position understanding

### 3. Training Strategy

-   **Primary Loss**: Cross-entropy on move prediction
-   **Auxiliary Loss**: Binary cross-entropy on game outcome
-   **Evaluation**: Top-1, Top-3, and Top-5 accuracy

## Data Splitting Best Practices

### Recommended Split Strategy

```
Total Data: 100%
├── Training: 70%
├── Validation: 20%
└── Test: 10%
```

### Important Considerations

1. **Game-Level Splitting**: Split by games, not individual positions

    - Prevents data leakage from same game
    - Better evaluation of true generalization

2. **Stratified Splitting**: Maintain outcome distribution

    - Ensure train/val/test have similar win/loss/draw ratios
    - Prevents bias toward certain game outcomes

3. **Temporal Splitting** (Optional):

    - Use chronological order for more realistic evaluation
    - Train on older games, test on recent ones
    - Captures evolution of Magnus's style over time

4. **Opening-Based Splitting** (Advanced):
    - Ensure diverse openings in all splits
    - Prevents overfitting to specific opening patterns

## Training Process

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv magnus_env
source magnus_env/bin/activate  # On macOS/Linux
# magnus_env\Scripts\activate   # On Windows

# Install dependencies
pip install -r training_requirements.txt
```

### Step 2: Data Preparation

```python
from improved_magnus_training import TrainingConfig, MagnusTrainer

# Configure training
config = TrainingConfig()
config.max_games = None  # Use all games
config.batch_size = 512
config.num_epochs = 100

# Create trainer
trainer = MagnusTrainer(config)
```

### Step 3: Training

```python
# Start training
model, history = trainer.train()
```

### Step 4: Evaluation

The trainer automatically evaluates on the test set and saves:

-   Model checkpoints
-   Training curves
-   Test set performance
-   Move vocabulary

## Expected Performance

### Realistic Expectations

-   **Top-1 Accuracy**: 35-45% (predicting exact Magnus move)
-   **Top-3 Accuracy**: 60-70% (Magnus move in top 3 predictions)
-   **Top-5 Accuracy**: 75-85% (Magnus move in top 5 predictions)

### Performance Comparison

-   **Strong Amateur**: ~25% top-1 accuracy
-   **Master Level**: ~35% top-1 accuracy
-   **Super-GM Level**: ~45% top-1 accuracy

## Advanced Training Techniques

### 1. Position Evaluation Integration

```python
# Add Stockfish evaluations as features
import chess.engine

def add_engine_evaluation(board):
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        return info["score"].relative.score()
```

### 2. Time Control Awareness

```python
# Include time pressure in features
def extract_time_features(game_node):
    clock = game_node.clock()
    if clock:
        return [clock, clock < 60, clock < 30]  # Time, low time flags
    return [0, False, False]
```

### 3. Style Transfer Learning

```python
# Pre-train on all strong players, fine-tune on Magnus
class StyleTransferTraining:
    def pretrain_on_strong_players(self):
        # Train on games from multiple strong players
        pass

    def finetune_on_magnus(self):
        # Fine-tune specifically on Magnus games
        pass
```

## Google Colab Training Setup

### Colab Notebook Template

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install python-chess torch torchvision scikit-learn matplotlib seaborn tqdm

# Upload PGN file to Colab
from google.colab import files
uploaded = files.upload()  # Upload carlsen-games.pgn

# Run training
from improved_magnus_training import TrainingConfig, MagnusTrainer

config = TrainingConfig()
config.device = "cuda"  # Use Colab GPU
config.batch_size = 1024  # Larger batch for A100
config.num_epochs = 200

trainer = MagnusTrainer(config)
model, history = trainer.train()

# Download trained model
files.download('/content/models/magnus_style_model/best_model.pth')
```

### Colab Resource Optimization

```python
# Enable mixed precision training
config.use_mixed_precision = True

# Gradient accumulation for large effective batch size
config.gradient_accumulation_steps = 4

# Memory optimization
config.pin_memory = True
config.num_workers = 2
```

## Cross-Validation Strategy

### Time-Series Cross-Validation

```python
def temporal_cross_validation(games, n_folds=5):
    # Sort games by date
    games_sorted = sorted(games, key=lambda g: g.headers.get('Date', ''))

    fold_size = len(games_sorted) // n_folds

    for i in range(n_folds):
        # Use current fold as test, previous as train
        test_start = i * fold_size
        test_end = (i + 1) * fold_size

        train_games = games_sorted[:test_start]
        test_games = games_sorted[test_start:test_end]

        yield train_games, test_games
```

### Opening-Stratified Cross-Validation

```python
def opening_stratified_cv(games, n_folds=5):
    # Group games by opening (first 4 moves)
    opening_groups = {}
    for game in games:
        opening = get_opening_signature(game)
        if opening not in opening_groups:
            opening_groups[opening] = []
        opening_groups[opening].append(game)

    # Distribute each opening across folds
    folds = [[] for _ in range(n_folds)]
    for opening, opening_games in opening_groups.items():
        for i, game in enumerate(opening_games):
            folds[i % n_folds].append(game)

    return folds
```

## Model Deployment

### Integration with Existing System

```python
# Update main.py to use Magnus model
class MagnusStyleAPI:
    def __init__(self):
        self.predictor = MagnusStylePredictor("models/magnus_style_model")
        self.lc0_engine = LC0Engine()  # Keep LC0 for comparison

    async def predict_move(self, fen: str):
        board = chess.Board(fen)

        # Get Magnus-style predictions
        magnus_moves = self.predictor.predict_move(board, top_k=3)

        # Get LC0 predictions for comparison
        lc0_moves = self.lc0_engine.get_top_moves(board, count=3)

        return {
            "magnus_style": magnus_moves,
            "engine_analysis": lc0_moves,
            "explanation": "Moves in Magnus Carlsen's style"
        }
```

## Monitoring and Evaluation

### Training Metrics

-   **Loss Curves**: Monitor for overfitting
-   **Accuracy Trends**: Track top-1, top-3, top-5 accuracy
-   **Learning Rate**: Adjust based on plateau detection

### Validation Strategies

```python
def evaluate_opening_performance(model, test_data):
    """Evaluate performance by opening type"""
    opening_performance = {}

    for position, move, opening in test_data:
        prediction = model.predict(position)
        correct = prediction == move

        if opening not in opening_performance:
            opening_performance[opening] = []
        opening_performance[opening].append(correct)

    return {opening: np.mean(results)
            for opening, results in opening_performance.items()}
```

### A/B Testing Framework

```python
def compare_models(magnus_model, baseline_model, test_positions):
    """Compare Magnus model against baseline"""
    results = {
        'magnus_better': 0,
        'baseline_better': 0,
        'tie': 0
    }

    for position in test_positions:
        magnus_pred = magnus_model.predict(position)
        baseline_pred = baseline_model.predict(position)

        # Compare against known strong moves
        magnus_score = evaluate_move_strength(position, magnus_pred)
        baseline_score = evaluate_move_strength(position, baseline_pred)

        if magnus_score > baseline_score:
            results['magnus_better'] += 1
        elif baseline_score > magnus_score:
            results['baseline_better'] += 1
        else:
            results['tie'] += 1

    return results
```

## Troubleshooting

### Common Issues

1. **Memory Errors**

    - Reduce batch size
    - Use gradient accumulation
    - Enable mixed precision training

2. **Poor Convergence**

    - Lower learning rate
    - Add learning rate scheduling
    - Increase model capacity

3. **Overfitting**

    - Add dropout layers
    - Use data augmentation (board rotations)
    - Early stopping

4. **Low Accuracy**
    - Check data quality
    - Verify move encoding
    - Add more features

### Performance Optimization

```python
# Data loading optimization
config.num_workers = 4
config.pin_memory = True
config.persistent_workers = True

# Model optimization
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
```

## Next Steps

1. **Start with basic training** using the provided code
2. **Evaluate initial results** and identify weaknesses
3. **Add advanced features** like position evaluation
4. **Experiment with architectures** (transformers, attention)
5. **Deploy and integrate** with your existing system

This approach gives you a practical, trainable Magnus Carlsen style model that can be deployed alongside your LC0 engine for comparison and demonstration.
