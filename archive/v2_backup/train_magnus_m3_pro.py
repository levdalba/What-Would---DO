#!/usr/bin/env python3
"""
Magnus Carlsen Neural Network Training for M3 Pro (Training Only)

This script loads pre-extracted positions and trains the Magnus Carlsen style
neural network using M3 Pro GPU acceleration. Position extraction must be
completed first using extract_positions_m3_pro.py.
"""

import sys
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from stockfish_magnus_trainer import (
    StockfishConfig,
    MagnusDataset,
    MagnusStyleModel,
    plot_training_curves,
)


def load_extracted_positions():
    """Load pre-extracted positions from pickle file"""
    data_path = Path("magnus_extracted_positions_m3_pro.pkl")

    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ Pre-extracted positions not found: {data_path}\n"
            f"   Please run extract_positions_m3_pro.py first!"
        )

    print(f"📂 Loading pre-extracted positions from {data_path}...")
    print(f"   File size: {data_path.stat().st_size / (1024*1024):.1f} MB")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Loaded {len(data['positions']):,} positions")
    print(f"   Extraction time: {data['extraction_time']/3600:.1f} hours")
    print(f"   Source: {data['metadata']['pgn_file']}")

    return data


def train_magnus_model():
    """Train Magnus Carlsen style model using pre-extracted positions"""

    # 🚀 M3 PRO GPU TRAINING CONFIGURATION
    print("🚀 M3 PRO MAGNUS TRAINING (GPU ACCELERATED)")
    print("=" * 60)

    # Load pre-extracted data
    extracted_data = load_extracted_positions()

    positions = extracted_data["positions"]
    features = extracted_data["features"]
    sf_moves = extracted_data["stockfish_moves"]
    magnus_moves = extracted_data["magnus_moves"]
    evaluations = extracted_data["evaluations"]

    # Training configuration optimized for M3 Pro GPU
    config = StockfishConfig()

    # GPU settings
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # M3 Pro GPU optimized settings
    if device.type == "mps":
        config.batch_size = 512  # Larger batches with GPU
        config.learning_rate = 0.001
        config.num_epochs = 50
        print(f"🎮 Using M3 Pro GPU (Metal Performance Shaders)")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Expected training time: ~1.3 hours")
    else:
        config.batch_size = 256  # Fallback for CPU
        config.learning_rate = 0.001
        config.num_epochs = 50
        print(f"🖥️  Using M3 Pro CPU")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Expected training time: ~2.5 hours")

    config.validation_split = 0.2
    config.test_split = 0.1
    config.early_stopping_patience = 8
    config.device = str(device)

    print(f"✅ Device: {device}")
    print(f"✅ Training positions: {len(positions):,}")
    print(f"✅ Epochs: {config.num_epochs}")
    print()

    # Create datasets
    print("📊 Creating train/validation/test datasets...")

    # Split data
    test_size = config.test_split
    val_size = config.validation_split / (1 - test_size)

    data = list(zip(positions, features, sf_moves, magnus_moves, evaluations))

    data_train_val, data_test = train_test_split(
        data, test_size=test_size, random_state=42
    )
    data_train, data_val = train_test_split(
        data_train_val, test_size=val_size, random_state=42
    )

    def unpack_data(split_data):
        return list(zip(*split_data))

    train_dataset = MagnusDataset(*unpack_data(data_train))
    val_dataset = MagnusDataset(*unpack_data(data_val))
    test_dataset = MagnusDataset(*unpack_data(data_test))

    # Share vocabulary
    val_dataset.move_to_idx = train_dataset.move_to_idx
    val_dataset.idx_to_move = train_dataset.idx_to_move
    val_dataset.vocab_size = train_dataset.vocab_size

    test_dataset.move_to_idx = train_dataset.move_to_idx
    test_dataset.idx_to_move = train_dataset.idx_to_move
    test_dataset.vocab_size = train_dataset.vocab_size

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "mps"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "mps"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "mps"),
    )

    print(f"📚 Dataset split:")
    print(f"   Train: {len(train_dataset):,} positions")
    print(f"   Validation: {len(val_dataset):,} positions")
    print(f"   Test: {len(test_dataset):,} positions")
    print(f"   Move vocabulary: {train_dataset.vocab_size:,} unique moves")
    print()

    # Create model
    model = MagnusStyleModel(
        config, train_dataset.vocab_size, len(train_dataset.feature_names)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model created:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Memory usage: ~{total_params * 4 / (1024*1024):.1f} MB")
    print()

    # Training setup
    move_criterion = nn.CrossEntropyLoss()
    eval_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_move_acc": [],
        "val_move_acc": [],
        "train_eval_loss": [],
        "val_eval_loss": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    print("🔥 Starting Magnus Carlsen training...")
    training_start_time = time.time()

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_move_correct = 0
        train_eval_losses = []
        train_total = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1:2d}/{config.num_epochs}", leave=False
        )

        for batch in train_pbar:
            optimizer.zero_grad()

            positions = batch["position"].to(device)
            features = batch["features"].to(device)
            magnus_moves = batch["magnus_move"].squeeze().to(device)
            evaluations = batch["evaluation"].squeeze().to(device)

            # Forward pass
            move_logits, eval_adjustment = model(positions, features)

            # Losses
            move_loss = move_criterion(move_logits, magnus_moves)
            eval_loss = eval_criterion(eval_adjustment.squeeze(), evaluations / 1000.0)
            total_loss = move_loss + 0.1 * eval_loss

            # Backward
            total_loss.backward()
            optimizer.step()

            # Metrics
            train_losses.append(total_loss.item())
            train_eval_losses.append(eval_loss.item())

            _, predicted = torch.max(move_logits, 1)
            train_move_correct += (predicted == magnus_moves).sum().item()
            train_total += magnus_moves.size(0)

            # Update progress bar
            current_acc = train_move_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "acc": f"{current_acc:.3f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        # Validation phase
        model.eval()
        val_losses = []
        val_move_correct = 0
        val_eval_losses = []
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                positions = batch["position"].to(device)
                features = batch["features"].to(device)
                magnus_moves = batch["magnus_move"].squeeze().to(device)
                evaluations = batch["evaluation"].squeeze().to(device)

                move_logits, eval_adjustment = model(positions, features)

                move_loss = move_criterion(move_logits, magnus_moves)
                eval_loss = eval_criterion(
                    eval_adjustment.squeeze(), evaluations / 1000.0
                )
                total_loss = move_loss + 0.1 * eval_loss

                val_losses.append(total_loss.item())
                val_eval_losses.append(eval_loss.item())

                _, predicted = torch.max(move_logits, 1)
                val_move_correct += (predicted == magnus_moves).sum().item()
                val_total += magnus_moves.size(0)

        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_move_acc = train_move_correct / train_total
        val_move_acc = val_move_correct / val_total
        train_eval_loss = np.mean(train_eval_losses)
        val_eval_loss = np.mean(val_eval_losses)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_move_acc"].append(train_move_acc)
        history["val_move_acc"].append(val_move_acc)
        history["train_eval_loss"].append(train_eval_loss)
        history["val_eval_loss"].append(val_eval_loss)

        # Calculate time estimates
        elapsed_time = time.time() - training_start_time
        epochs_completed = epoch + 1
        avg_time_per_epoch = elapsed_time / epochs_completed
        estimated_total_time = avg_time_per_epoch * config.num_epochs
        eta = estimated_total_time - elapsed_time

        # Log epoch results
        print(
            f"Epoch {epoch+1:2d}: "
            f"Loss {val_loss:.4f} | "
            f"Acc {val_move_acc:.3f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"ETA {eta/60:.0f}m"
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save model
            model_save_dir = Path("models/magnus_m3_pro_trained")
            model_save_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_move_acc": val_move_acc,
                "config": config.__dict__,
                "dataset_info": {
                    "vocab_size": train_dataset.vocab_size,
                    "move_to_idx": train_dataset.move_to_idx,
                    "feature_names": train_dataset.feature_names,
                    "train_size": len(train_dataset),
                    "val_size": len(val_dataset),
                    "test_size": len(test_dataset),
                },
                "training_metadata": {
                    "hardware": "M3 Pro",
                    "device": str(device),
                    "total_positions": len(positions),
                    "training_start": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "extraction_time_hours": extracted_data["extraction_time"] / 3600,
                },
            }

            model_path = model_save_dir / "best_magnus_model.pth"
            torch.save(checkpoint, model_path)
            print(f"         ✅ New best model saved!")

        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"         🛑 Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - training_start_time

    # Final evaluation
    print(f"\n🧪 Final evaluation on test set...")
    model.eval()
    test_move_correct = 0
    test_total = 0
    eval_errors = []

    with torch.no_grad():
        for batch in test_loader:
            positions = batch["position"].to(device)
            features = batch["features"].to(device)
            magnus_moves = batch["magnus_move"].squeeze().to(device)
            evaluations = batch["evaluation"].squeeze().to(device)

            move_logits, eval_adjustment = model(positions, features)

            _, predicted = torch.max(move_logits, 1)
            test_move_correct += (predicted == magnus_moves).sum().item()
            test_total += magnus_moves.size(0)

            eval_pred = eval_adjustment.squeeze() * 1000.0
            eval_errors.extend(torch.abs(eval_pred - evaluations).cpu().numpy())

    test_acc = test_move_correct / test_total
    mean_eval_error = np.mean(eval_errors)

    # Save training history and results
    model_save_dir = Path("models/magnus_m3_pro_trained")

    history_path = model_save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    test_results = {
        "move_accuracy": float(test_acc),
        "mean_eval_error": float(mean_eval_error),
        "total_samples": int(test_total),
        "training_time_hours": training_time / 3600,
        "total_project_time_hours": (extracted_data["extraction_time"] + training_time)
        / 3600,
        "hardware": "M3 Pro",
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = model_save_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    # Create training curves
    plot_training_curves(history, str(model_save_dir))

    # Final results
    print(f"\n🎉 MAGNUS CARLSEN TRAINING COMPLETED!")
    print(f"=" * 60)
    print(f"📊 FINAL RESULTS:")
    print(f"   🎯 Move Accuracy: {test_acc:.1%}")
    print(f"   📈 Evaluation Error: {mean_eval_error:.0f} centipawns")
    print(f"   📚 Test Samples: {test_total:,}")
    print(f"   ⏱️  Training Time: {training_time/3600:.1f} hours")
    print(
        f"   🏁 Total Project Time: {(extracted_data['extraction_time'] + training_time)/3600:.1f} hours"
    )
    print(f"   🎮 Device Used: {device}")
    print(f"   💾 Model Saved: {model_path}")
    print(f"")
    print(f"🏆 Your Magnus Carlsen-style chess engine is ready!")
    print(f"   The model can now predict Magnus's moves with {test_acc:.1%} accuracy!")

    return model, history, test_results


if __name__ == "__main__":
    try:
        model, history, results = train_magnus_model()
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
