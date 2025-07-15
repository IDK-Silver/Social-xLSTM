#!/usr/bin/env python3
"""
Minimal training test to verify overfitting fix.

This script runs a minimal training session to quickly test if the overfitting
fix is working without needing the full training infrastructure.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from social_xlstm.dataset.core.timeseries import TrafficTimeSeries
from social_xlstm.dataset.config import TrafficDatasetConfig
from social_xlstm.models.lstm import TrafficLSTM, TrafficLSTMConfig


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_config(yaml_config):
    """Create TrafficDatasetConfig from YAML config."""
    return TrafficDatasetConfig(
        hdf5_path=yaml_config['h5_file_path'],
        selected_vdids=yaml_config.get('selected_vdids', [yaml_config['select_vd_id']]),
        sequence_length=yaml_config['dataset']['sequence_length'],
        prediction_length=yaml_config['dataset']['prediction_length'],
        train_ratio=yaml_config['dataset']['train_ratio'],
        val_ratio=yaml_config['dataset']['val_ratio'],
        test_ratio=yaml_config['dataset'].get('test_ratio', 0.0),
        normalize=yaml_config['dataset']['normalize'],
        normalization_method=yaml_config['dataset']['normalization_method']
    )


def create_model_config(yaml_config):
    """Create TrafficLSTMConfig from YAML config."""
    # For multi-VD training: 3 VDs * 5 features = 15 input features
    num_vds = len(yaml_config.get('selected_vdids', [yaml_config['select_vd_id']]))
    num_features = 5
    input_size = num_vds * num_features
    output_size = num_vds * num_features
    
    return TrafficLSTMConfig(
        input_size=input_size,  # VDs * features
        hidden_size=yaml_config['model']['hidden_size'],
        num_layers=yaml_config['model']['num_layers'],
        output_size=output_size,  # Same as input for prediction
        dropout=yaml_config['model']['dropout'],
        bidirectional=yaml_config['model']['bidirectional']
    )


def run_minimal_training(config_path, max_epochs=10):
    """Run a minimal training session."""
    print(f"🚀 MINIMAL TRAINING TEST")
    print(f"{'='*40}")
    print(f"Config: {config_path}")
    print(f"Max epochs: {max_epochs}")
    
    # Load configuration
    yaml_config = load_config(config_path)
    dataset_config = create_dataset_config(yaml_config)
    model_config = create_model_config(yaml_config)
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   HDF5 path: {dataset_config.hdf5_path}")
    print(f"   Selected VDs: {dataset_config.selected_vdids}")
    print(f"   Sequence length: {dataset_config.sequence_length}")
    
    print(f"\n🧠 Model Configuration:")
    print(f"   Hidden size: {model_config.hidden_size}")
    print(f"   Num layers: {model_config.num_layers}")
    print(f"   Dropout: {model_config.dropout}")
    
    # Create datasets
    try:
        print(f"\n📂 Loading datasets...")
        train_dataset = TrafficTimeSeries(dataset_config, split='train')
        val_dataset = TrafficTimeSeries(dataset_config, split='val', scaler=train_dataset.get_scaler())
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"❌ Dataset is empty!")
            return None
            
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=yaml_config['training']['batch_size'],
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=yaml_config['training']['batch_size'],
        shuffle=False
    )
    
    # Create model
    model = TrafficLSTM(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Device: {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=yaml_config['training']['learning_rate'],
        weight_decay=yaml_config['training']['weight_decay']
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\n🔄 Training Progress:")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Get data
            input_seq = batch['input_seq'].to(device)  # [batch, seq, vd, features]
            target_seq = batch['target_seq'].to(device)  # [batch, pred, vd, features]
            
            # Reshape for model: combine VD and feature dimensions
            batch_size, seq_len, num_vds, num_features = input_seq.shape
            input_reshaped = input_seq.view(batch_size, seq_len, -1)  # [batch, seq, vd*features]
            
            # Forward pass
            outputs = model(input_reshaped)  # [batch, pred, vd*features]
            
            # Reshape target to match output
            target_reshaped = target_seq.view(batch_size, -1, num_vds * num_features)
            
            # Calculate loss
            loss = criterion(outputs, target_reshaped)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), yaml_config['training']['gradient_clip_value'])
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_seq = batch['input_seq'].to(device)
                target_seq = batch['target_seq'].to(device)
                
                # Reshape
                batch_size, seq_len, num_vds, num_features = input_seq.shape
                input_reshaped = input_seq.view(batch_size, seq_len, -1)
                target_reshaped = target_seq.view(batch_size, -1, num_vds * num_features)
                
                # Forward pass
                outputs = model(input_reshaped)
                loss = criterion(outputs, target_reshaped)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
        
        # Record losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        overfitting_ratio = avg_val_loss / avg_train_loss if avg_train_loss > 0 else float('inf')
        print(f"   Epoch {epoch+1:2d}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, ratio={overfitting_ratio:.2f}")
    
    # Final analysis
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    final_overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
    
    results = {
        'total_epochs': max_epochs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_overfitting_ratio': final_overfitting_ratio,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    print(f"\n📊 Training Results Summary:")
    print(f"   Best epoch: {best_epoch}/{max_epochs}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Final train loss: {final_train_loss:.4f}")
    print(f"   Final val loss: {final_val_loss:.4f}")
    print(f"   Final overfitting ratio: {final_overfitting_ratio:.2f}")
    
    # Assessment
    if final_overfitting_ratio < 3:
        print(f"   🎉 EXCELLENT: No significant overfitting!")
    elif final_overfitting_ratio < 8:
        print(f"   ✅ GOOD: Minimal overfitting (much improved from 98+)")
    elif final_overfitting_ratio < 20:
        print(f"   ⚠️ MODERATE: Some overfitting, but improved")
    else:
        print(f"   ❌ POOR: Still significant overfitting")
    
    return results


def create_training_plot(results, output_path="blob/debug/minimal_training_test.png"):
    """Create training curve plot."""
    if not results:
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(len(results['train_losses']))
    
    # Plot 1: Training curves
    ax1.plot(epochs, results['train_losses'], 'b-', label='Train Loss', alpha=0.8)
    ax1.plot(epochs, results['val_losses'], 'r-', label='Val Loss', alpha=0.8)
    
    # Mark best epoch
    best_epoch = results['best_epoch']
    if best_epoch < len(results['val_losses']):
        ax1.scatter(best_epoch, results['val_losses'][best_epoch], 
                   color='red', s=100, marker='*', zorder=5, label='Best Epoch')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Curves (Fixed Model)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overfitting ratio over time
    overfitting_ratios = [val/train if train > 0 else 0 
                         for train, val in zip(results['train_losses'], results['val_losses'])]
    
    ax2.plot(epochs, overfitting_ratios, 'g-', label='Val/Train Ratio', alpha=0.8)
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Good Threshold (5)')
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Acceptable Threshold (10)')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss / Train Loss')
    ax2.set_title('Overfitting Ratio Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(20, max(overfitting_ratios) * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plot saved to: {output_path}")


def main():
    """Main test execution."""
    print(f"🧪 MINIMAL OVERFITTING FIX TEST")
    print(f"{'='*60}")
    
    config_path = "cfgs/fixed/lstm_fixed.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print(f"   Make sure you've run the fix generation script first")
        return
    
    try:
        # Run training test
        results = run_minimal_training(config_path, max_epochs=8)
        
        if results:
            # Create visualization
            create_training_plot(results)
            
            # Save results
            results_path = "blob/debug/minimal_training_results.json"
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved to: {results_path}")
            
            # Final comparison
            print(f"\n🔍 COMPARISON WITH ORIGINAL PROBLEM:")
            print(f"   Original train/val ratio: 98+ (severe overfitting)")
            print(f"   Fixed train/val ratio: {results['final_overfitting_ratio']:.2f}")
            
            improvement = 98 / results['final_overfitting_ratio'] if results['final_overfitting_ratio'] > 0 else float('inf')
            print(f"   Improvement factor: {improvement:.1f}x better")
            
            if results['final_overfitting_ratio'] < 8:
                print(f"\n🎉 SUCCESS: Overfitting fix is working!")
                print(f"✅ Ready to proceed with full training")
            else:
                print(f"\n⚠️ PARTIAL SUCCESS: Improved but may need more tuning")
                print(f"💡 Consider: more dropout, smaller model, or more data")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()