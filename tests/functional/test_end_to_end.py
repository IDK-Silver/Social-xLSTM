"""
Functional tests for end-to-end workflows.

These tests verify that the complete system works together
as expected from a user's perspective.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

from social_xlstm.models.lstm import TrafficLSTM
from social_xlstm.training.recorder import TrainingRecorder
from social_xlstm.visualization.training_visualizer import TrainingVisualizer


@pytest.mark.functional
class TestEndToEndWorkflows:
    """Functional tests for complete user workflows."""
    
    @pytest.fixture
    def functional_temp_dir(self):
        """Create temporary directory for functional tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_baseline_experiment_workflow(self, functional_temp_dir):
        """Test complete baseline experiment workflow."""
        
        # Step 1: Create and train single VD model
        print("Step 1: Creating single VD model...")
        single_vd_model = TrafficLSTM.create_single_vd_model(
            input_size=3,
            hidden_size=64,
            num_layers=2
        )
        
        # Step 2: Create recorder for single VD
        single_vd_recorder = TrainingRecorder(
            experiment_name="single_vd_baseline",
            model_config=single_vd_model.config.__dict__,
            training_config={'epochs': 10, 'batch_size': 32}
        )
        
        # Step 3: Simulate training
        print("Step 3: Simulating training...")
        for epoch in range(10):
            single_vd_recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 * np.exp(-0.1 * epoch) + 0.01 * np.random.rand(),
                val_loss=1.1 * np.exp(-0.08 * epoch) + 0.02 * np.random.rand(),
                train_metrics={
                    'mae': 0.5 * np.exp(-0.1 * epoch),
                    'mse': 0.25 * np.exp(-0.1 * epoch),
                    'rmse': 0.5 * np.exp(-0.1 * epoch)
                },
                val_metrics={
                    'mae': 0.55 * np.exp(-0.08 * epoch),
                    'mse': 0.3 * np.exp(-0.08 * epoch),
                    'rmse': 0.55 * np.exp(-0.08 * epoch)
                },
                learning_rate=0.001 * (0.9 ** epoch),
                epoch_time=30.0 + 5 * np.random.rand()
            )
        
        # Step 4: Create and train multi VD model
        print("Step 4: Creating multi VD model...")
        multi_vd_model = TrafficLSTM.create_multi_vd_model(
            num_vds=5,
            input_size=3,
            hidden_size=64,
            num_layers=2
        )
        
        # Step 5: Create recorder for multi VD
        multi_vd_recorder = TrainingRecorder(
            experiment_name="multi_vd_baseline",
            model_config=multi_vd_model.config.__dict__,
            training_config={'epochs': 10, 'batch_size': 16}
        )
        
        # Step 6: Simulate multi VD training
        print("Step 6: Simulating multi VD training...")
        for epoch in range(10):
            multi_vd_recorder.log_epoch(
                epoch=epoch,
                train_loss=0.8 * np.exp(-0.12 * epoch) + 0.01 * np.random.rand(),
                val_loss=0.9 * np.exp(-0.1 * epoch) + 0.02 * np.random.rand(),
                train_metrics={
                    'mae': 0.4 * np.exp(-0.12 * epoch),
                    'mse': 0.16 * np.exp(-0.12 * epoch),
                    'rmse': 0.4 * np.exp(-0.12 * epoch)
                },
                val_metrics={
                    'mae': 0.45 * np.exp(-0.1 * epoch),
                    'mse': 0.2 * np.exp(-0.1 * epoch),
                    'rmse': 0.45 * np.exp(-0.1 * epoch)
                },
                learning_rate=0.0008 * (0.9 ** epoch),
                epoch_time=45.0 + 10 * np.random.rand()
            )
        
        # Step 7: Save experiment results
        print("Step 7: Saving experiment results...")
        experiment_dir = functional_temp_dir / "baseline_experiments"
        experiment_dir.mkdir()
        
        # Save single VD results
        single_vd_dir = experiment_dir / "single_vd"
        single_vd_dir.mkdir()
        
        single_vd_recorder.save(single_vd_dir / "training_record.json")
        single_vd_recorder.export_to_csv(single_vd_dir / "training_history.csv")
        
        # Save multi VD results  
        multi_vd_dir = experiment_dir / "multi_vd"
        multi_vd_dir.mkdir()
        
        multi_vd_recorder.save(multi_vd_dir / "training_record.json")
        multi_vd_recorder.export_to_csv(multi_vd_dir / "training_history.csv")
        
        # Step 8: Generate visualizations
        print("Step 8: Generating visualizations...")
        visualizer = TrainingVisualizer()
        
        # Single VD visualizations
        visualizer.create_training_report(
            single_vd_recorder, 
            single_vd_dir / "report"
        )
        
        # Multi VD visualizations
        visualizer.create_training_report(
            multi_vd_recorder,
            multi_vd_dir / "report"
        )
        
        # Comparison visualization
        comparison_fig = visualizer.plot_experiment_comparison(
            [single_vd_recorder, multi_vd_recorder],
            save_path=experiment_dir / "comparison.png"
        )
        
        # Step 9: Verify all outputs exist
        print("Step 9: Verifying outputs...")
        
        # Check single VD outputs
        assert (single_vd_dir / "training_record.json").exists()
        assert (single_vd_dir / "training_history.csv").exists()
        assert (single_vd_dir / "report").exists()
        assert (single_vd_dir / "report" / "training_summary.txt").exists()
        
        # Check multi VD outputs
        assert (multi_vd_dir / "training_record.json").exists()
        assert (multi_vd_dir / "training_history.csv").exists()
        assert (multi_vd_dir / "report").exists()
        assert (multi_vd_dir / "report" / "training_summary.txt").exists()
        
        # Check comparison
        assert (experiment_dir / "comparison.png").exists()
        
        # Step 10: Test loading and analysis
        print("Step 10: Testing loading and analysis...")
        
        # Load experiments
        loaded_single = TrainingRecorder.load(single_vd_dir / "training_record.json")
        loaded_multi = TrainingRecorder.load(multi_vd_dir / "training_record.json")
        
        # Compare results
        comparison = loaded_single.compare_with(loaded_multi)
        
        # Verify comparison results
        assert 'experiment_names' in comparison
        assert 'best_val_loss' in comparison
        assert 'total_epochs' in comparison
        
        # Multi VD should perform better (lower loss)
        single_best = loaded_single.get_best_epoch().val_loss
        multi_best = loaded_multi.get_best_epoch().val_loss
        
        print(f"Single VD best val loss: {single_best:.4f}")
        print(f"Multi VD best val loss: {multi_best:.4f}")
        
        # In this simulation, multi VD should be better
        assert multi_best < single_best
        
        print("✅ Baseline experiment workflow completed successfully!")
    
    def test_new_metric_calculation_workflow(self, functional_temp_dir):
        """Test workflow for calculating new metrics post-training."""
        
        # Step 1: Load existing experiment
        recorder = TrainingRecorder(
            experiment_name="metric_test",
            model_config={'input_size': 3},
            training_config={'epochs': 5}
        )
        
        # Add some training data
        for epoch in range(5):
            recorder.log_epoch(
                epoch=epoch,
                train_loss=1.0 / (epoch + 1),
                val_loss=1.1 / (epoch + 1),
                train_metrics={'mae': 0.1 / (epoch + 1), 'mse': 0.01 / (epoch + 1)},
                val_metrics={'mae': 0.11 / (epoch + 1), 'mse': 0.011 / (epoch + 1)},
                learning_rate=0.001,
                epoch_time=30.0
            )
        
        # Save experiment
        save_path = functional_temp_dir / "experiment.json"
        recorder.save(save_path)
        
        # Step 2: Load experiment later
        loaded_recorder = TrainingRecorder.load(save_path)
        
        # Step 3: Calculate new custom metrics
        def custom_metric_1(epoch_record):
            """Custom metric: ratio of val_loss to train_loss"""
            if epoch_record.val_loss and epoch_record.train_loss:
                return epoch_record.val_loss / epoch_record.train_loss
            return None
        
        def custom_metric_2(epoch_record):
            """Custom metric: improvement rate"""
            if epoch_record.epoch == 0:
                return 0.0
            return (1.0 - epoch_record.train_loss) * 100  # Improvement percentage
        
        # Calculate custom metrics for all epochs
        custom_metrics_1 = []
        custom_metrics_2 = []
        
        for epoch_record in loaded_recorder.epochs:
            custom_metrics_1.append(custom_metric_1(epoch_record))
            custom_metrics_2.append(custom_metric_2(epoch_record))
        
        # Step 4: Verify custom metrics
        assert len(custom_metrics_1) == 5
        assert len(custom_metrics_2) == 5
        
        # All ratios should be > 1 (val_loss > train_loss)
        assert all(ratio > 1.0 for ratio in custom_metrics_1 if ratio is not None)
        
        # Improvement should increase over time
        assert custom_metrics_2[-1] > custom_metrics_2[0]
        
        # Step 5: Create visualization with new metrics
        # This demonstrates how new metrics can be added post-training
        visualizer = TrainingVisualizer()
        
        # Create custom plot
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(custom_metrics_1) + 1)
        
        # Plot custom metric 1
        axes[0].plot(epochs, custom_metrics_1, 'o-', label='Val/Train Ratio')
        axes[0].set_title('Custom Metric 1: Val/Train Loss Ratio')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Ratio')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot custom metric 2
        axes[1].plot(epochs, custom_metrics_2, 's-', label='Improvement %', color='green')
        axes[1].set_title('Custom Metric 2: Improvement Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Improvement %')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        custom_plot_path = functional_temp_dir / "custom_metrics.png"
        plt.savefig(custom_plot_path)
        plt.close()
        
        # Verify custom plot was created
        assert custom_plot_path.exists()
        
        print("✅ New metric calculation workflow completed successfully!")
    
    @pytest.mark.slow
    def test_experiment_comparison_workflow(self, functional_temp_dir):
        """Test workflow for comparing multiple experiments."""
        
        # Create multiple experiments with different configurations
        experiments = []
        
        configs = [
            {'name': 'small_model', 'hidden_size': 32, 'num_layers': 1},
            {'name': 'medium_model', 'hidden_size': 64, 'num_layers': 2},
            {'name': 'large_model', 'hidden_size': 128, 'num_layers': 3}
        ]
        
        for config in configs:
            recorder = TrainingRecorder(
                experiment_name=config['name'],
                model_config={
                    'hidden_size': config['hidden_size'],
                    'num_layers': config['num_layers']
                },
                training_config={'epochs': 8, 'batch_size': 32}
            )
            
            # Simulate training with different performance
            base_loss = 1.0 - (config['hidden_size'] / 128) * 0.2  # Larger models perform better
            
            for epoch in range(8):
                recorder.log_epoch(
                    epoch=epoch,
                    train_loss=base_loss * np.exp(-0.1 * epoch),
                    val_loss=base_loss * 1.1 * np.exp(-0.08 * epoch),
                    train_metrics={'mae': base_loss * 0.5 * np.exp(-0.1 * epoch)},
                    val_metrics={'mae': base_loss * 0.55 * np.exp(-0.08 * epoch)},
                    learning_rate=0.001,
                    epoch_time=30.0 + config['num_layers'] * 10  # Deeper models take longer
                )
            
            experiments.append(recorder)
        
        # Save all experiments
        experiment_dir = functional_temp_dir / "comparison_experiments"
        experiment_dir.mkdir()
        
        for recorder in experiments:
            recorder.save(experiment_dir / f"{recorder.experiment_name}.json")
        
        # Load experiments
        loaded_experiments = []
        for config in configs:
            loaded_recorder = TrainingRecorder.load(
                experiment_dir / f"{config['name']}.json"
            )
            loaded_experiments.append(loaded_recorder)
        
        # Create comparison visualizations
        visualizer = TrainingVisualizer()
        
        # Multi-experiment comparison
        comparison_fig = visualizer.plot_experiment_comparison(
            loaded_experiments,
            save_path=experiment_dir / "model_comparison.png"
        )
        
        # Verify comparison plot
        assert (experiment_dir / "model_comparison.png").exists()
        
        # Analyze results
        results = []
        for recorder in loaded_experiments:
            best_epoch = recorder.get_best_epoch()
            summary = recorder.get_training_summary()
            
            results.append({
                'name': recorder.experiment_name,
                'best_val_loss': best_epoch.val_loss,
                'total_time': summary['total_time'],
                'avg_epoch_time': summary['avg_epoch_time'],
                'model_size': recorder.model_config['hidden_size']
            })
        
        # Sort by performance
        results.sort(key=lambda x: x['best_val_loss'])
        
        # Verify that larger models generally perform better
        assert results[0]['model_size'] >= results[-1]['model_size']
        
        print("✅ Experiment comparison workflow completed successfully!")
        print(f"Best model: {results[0]['name']} with val_loss: {results[0]['best_val_loss']:.4f}")
        
        return results