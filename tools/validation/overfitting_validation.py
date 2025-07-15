#!/usr/bin/env python3
"""
Test the overfitting fix by running a quick training comparison.

This script runs a quick training session with the fixed configuration
to verify that overfitting has been significantly reduced.
"""

import sys
import json
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def run_quick_training(config_path, experiment_name, max_epochs=10):
    """Run a quick training session to test overfitting behavior."""
    print(f"🚀 Running quick training test: {experiment_name}")
    print(f"   Config: {config_path}")
    print(f"   Max epochs: {max_epochs}")
    
    # Prepare training command (using existing training scripts)
    if "lstm" in config_path.lower():
        script_path = "scripts/train/without_social_pooling/train_single_vd.py"
    else:
        script_path = "scripts/train/without_social_pooling/train_single_vd.py"  # Will be adapted for xlstm
    
    cmd = [
        "python", script_path,
        "--config", config_path,
        "--max_epochs", str(max_epochs),
        "--experiment_name", experiment_name
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # Run training
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        end_time = time.time()
        
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Training completed in {training_time:.1f}s")
            return True, training_time, result.stdout
        else:
            print(f"   ❌ Training failed:")
            print(f"      stdout: {result.stdout}")
            print(f"      stderr: {result.stderr}")
            return False, training_time, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Training timed out after 10 minutes")
        return False, 600, "Timeout"
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return False, 0, str(e)


def analyze_training_results(experiment_dir):
    """Analyze training results to check overfitting improvement."""
    print(f"\n📊 Analyzing training results: {experiment_dir}")
    
    # Look for training history
    history_file = Path(experiment_dir) / "training_history.json"
    
    if not history_file.exists():
        print(f"   ❌ Training history not found: {history_file}")
        return None
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        epochs = history.get('epochs', [])
        if not epochs:
            print(f"   ❌ No epoch data found")
            return None
        
        # Extract key metrics
        train_losses = [epoch['train_loss'] for epoch in epochs]
        val_losses = [epoch['val_loss'] for epoch in epochs]
        
        best_epoch = history.get('best_epoch', 0)
        best_val_loss = history.get('best_val_loss', float('inf'))
        
        final_train_loss = train_losses[-1] if train_losses else 0
        final_val_loss = val_losses[-1] if val_losses else 0
        
        # Calculate overfitting metrics
        final_overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
        best_val_epoch_ratio = best_epoch / len(epochs) if len(epochs) > 0 else 0
        
        metrics = {
            'total_epochs': len(epochs),
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_overfitting_ratio': final_overfitting_ratio,
            'best_epoch_ratio': best_val_epoch_ratio,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"   📈 Training Summary:")
        print(f"      Total epochs: {metrics['total_epochs']}")
        print(f"      Best epoch: {metrics['best_epoch']} ({best_val_epoch_ratio:.1%} through training)")
        print(f"      Final train loss: {final_train_loss:.4f}")
        print(f"      Final val loss: {final_val_loss:.4f}")
        print(f"      Overfitting ratio: {final_overfitting_ratio:.2f}")
        
        # Assessment
        overfitting_assessment = ""
        if final_overfitting_ratio < 2:
            overfitting_assessment = "✅ EXCELLENT - No significant overfitting"
        elif final_overfitting_ratio < 5:
            overfitting_assessment = "✅ GOOD - Minimal overfitting"
        elif final_overfitting_ratio < 10:
            overfitting_assessment = "⚠️ MODERATE - Some overfitting"
        else:
            overfitting_assessment = "❌ SEVERE - Still overfitting"
        
        print(f"      Assessment: {overfitting_assessment}")
        
        return metrics
        
    except Exception as e:
        print(f"   ❌ Error analyzing results: {e}")
        return None


def create_comparison_plot(results, output_path="blob/debug/overfitting_fix_test.png"):
    """Create comparison plot showing before/after overfitting."""
    print(f"\n📊 Creating comparison plot")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training curves for fixed models
    ax1 = axes[0]
    
    colors = ['blue', 'red', 'green']
    for i, (name, metrics) in enumerate(results.items()):
        if metrics and 'train_losses' in metrics:
            epochs = range(len(metrics['train_losses']))
            ax1.plot(epochs, metrics['train_losses'], f'--', color=colors[i % len(colors)], 
                    alpha=0.7, label=f'{name} Train')
            ax1.plot(epochs, metrics['val_losses'], f'-', color=colors[i % len(colors)], 
                    alpha=0.9, label=f'{name} Val')
            
            # Mark best epoch
            best_epoch = metrics['best_epoch']
            if best_epoch < len(metrics['val_losses']):
                ax1.scatter(best_epoch, metrics['val_losses'][best_epoch], 
                           color=colors[i % len(colors)], s=100, marker='*', 
                           label=f'{name} Best')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Fixed Models Training Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overfitting comparison
    ax2 = axes[1]
    
    # Historical data (from previous analysis)
    historical_data = {
        'Original LSTM': {'ratio': 113.55, 'best_epoch_ratio': 0.0},
        'Original xLSTM': {'ratio': 98.98, 'best_epoch_ratio': 0.333}
    }
    
    # Current data
    current_data = {}
    for name, metrics in results.items():
        if metrics:
            current_data[f'Fixed {name}'] = {
                'ratio': metrics.get('final_overfitting_ratio', 0),
                'best_epoch_ratio': metrics.get('best_epoch_ratio', 0)
            }
    
    # Combined data
    all_data = {**historical_data, **current_data}
    
    names = list(all_data.keys())
    ratios = [all_data[name]['ratio'] for name in names]
    colors_bar = ['red', 'orange', 'green', 'lightgreen'][:len(names)]
    
    bars = ax2.bar(range(len(names)), ratios, color=colors_bar, alpha=0.7)
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('Final Train/Val Loss Ratio')
    ax2.set_title('Overfitting Comparison: Before vs After Fix')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    
    # Add quality threshold line
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Good Quality Threshold')
    ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        if height < 100:  # Only label reasonable values
            ax2.annotate(f'{ratio:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Plot saved to: {output_path}")


def main():
    """Main test execution."""
    print(f"🧪 OVERFITTING FIX EFFECTIVENESS TEST")
    print(f"{'='*60}")
    
    # Test configurations
    test_configs = [
        ("cfgs/fixed/lstm_fixed.yaml", "lstm_fix_test"),
        # ("cfgs/fixed/xlstm_fixed.yaml", "xlstm_fix_test")  # Skip for now to save time
    ]
    
    results = {}
    
    for config_path, experiment_name in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing: {experiment_name}")
        print(f"{'='*50}")
        
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            results[experiment_name] = None
            continue
        
        # Run quick training test
        success, training_time, output = run_quick_training(config_path, experiment_name, max_epochs=10)
        
        if success:
            # Analyze results
            experiment_dir = f"blob/experiments/dev/{experiment_name}"  # Assuming dev output
            metrics = analyze_training_results(experiment_dir)
            results[experiment_name] = metrics
        else:
            print(f"❌ Training failed for {experiment_name}")
            print(f"   Output: {output}")
            results[experiment_name] = None
    
    # Create comparison visualization
    create_comparison_plot(results)
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print(f"{'='*40}")
    
    successful_tests = [name for name, metrics in results.items() if metrics is not None]
    
    if successful_tests:
        print(f"✅ Successfully tested: {', '.join(successful_tests)}")
        
        for name, metrics in results.items():
            if metrics:
                ratio = metrics.get('final_overfitting_ratio', float('inf'))
                best_epoch_ratio = metrics.get('best_epoch_ratio', 0)
                
                print(f"\n📊 {name}:")
                print(f"   Train/Val ratio: {ratio:.2f} (target: <5)")
                print(f"   Best epoch: {metrics['best_epoch']}/{metrics['total_epochs']} ({best_epoch_ratio:.1%})")
                
                if ratio < 5 and best_epoch_ratio > 0.3:
                    print(f"   🎉 SUCCESS: Overfitting significantly reduced!")
                elif ratio < 10:
                    print(f"   ✅ IMPROVED: Much better than original (was 98+)")
                else:
                    print(f"   ⚠️ PARTIAL: Some improvement but may need more tuning")
        
        print(f"\n💡 Recommendations:")
        print(f"   1. If results are good: proceed with full training")
        print(f"   2. If still overfitting: increase dropout further or reduce model size")
        print(f"   3. Monitor training curves for stable learning")
        
    else:
        print(f"❌ No successful tests completed")
        print(f"💡 Debug training pipeline and try again")


if __name__ == "__main__":
    main()