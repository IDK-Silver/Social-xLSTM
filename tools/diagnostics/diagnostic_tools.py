#!/usr/bin/env python3
"""
Diagnostic tools - unified interface for dataset and model diagnostics.

This module provides a unified interface to the diagnostic functions
that have been integrated into the main codebase.

Usage:
    python scripts/utils/diagnostic_tools.py --h5_path dataset.h5 --vd_id VD-28-0740-000-001
    python scripts/utils/diagnostic_tools.py --comprehensive --h5_path dataset.h5 --vd_id VD-28-0740-000-001
"""

import argparse
import sys
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from social_xlstm.evaluation.evaluator import DatasetDiagnostics


def run_dataset_analysis(h5_path: str, vd_id: str) -> None:
    """Run dataset analysis using the integrated method."""
    print(f"🔍 DATASET ANALYSIS")
    print(f"{'='*50}")
    
    diagnostics = DatasetDiagnostics()
    results = diagnostics.analyze_h5_dataset(h5_path, vd_id)
    
    print(f"\n📊 Dataset: {h5_path}")
    print(f"📈 VD ID: {vd_id}")
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        if 'available_vds' in results:
            print(f"Available VDs: {results['available_vds']}")
        return
    
    print(f"\n✅ Analysis complete - {len(results)} metrics calculated")


def run_split_analysis(h5_path: str, vd_id: str) -> None:
    """Run data split analysis using the integrated method."""
    print(f"🔄 DATA SPLIT ANALYSIS")
    print(f"{'='*50}")
    
    diagnostics = DatasetDiagnostics()
    results = diagnostics.analyze_data_splits(h5_path, vd_id)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n✅ Split analysis complete")


def run_model_analysis() -> None:
    """Run model complexity analysis using the integrated method."""
    print(f"🧠 MODEL COMPLEXITY ANALYSIS")
    print(f"{'='*50}")
    
    diagnostics = DatasetDiagnostics()
    results = diagnostics.analyze_model_complexity()
    
    print(f"\n✅ Model analysis complete - {len(results)} models analyzed")


def create_plots(h5_path: str, vd_id: str, output_dir: str = "blob/debug") -> None:
    """Create diagnostic plots using the integrated method."""
    print(f"📈 CREATING DIAGNOSTIC PLOTS")
    print(f"{'='*50}")
    
    diagnostics = DatasetDiagnostics()
    success = diagnostics.create_diagnostic_plots(h5_path, vd_id, output_dir)
    
    if success:
        print(f"\n✅ Plots saved to: {output_dir}/")
    else:
        print(f"\n❌ Error creating plots")


def run_comprehensive_diagnosis(h5_path: str, vd_id: str, output_dir: str = "blob/debug") -> None:
    """Run comprehensive diagnosis using the integrated method."""
    print(f"🎯 COMPREHENSIVE DIAGNOSIS")
    print(f"{'='*50}")
    
    diagnostics = DatasetDiagnostics()
    results = diagnostics.comprehensive_diagnosis(h5_path, vd_id, output_dir=output_dir)
    
    print(f"\n📊 Diagnosis Results:")
    print(f"   Dataset: {results['h5_path']}")
    print(f"   VD ID: {results['vd_id']}")
    print(f"   Issues found: {len(results['issues_found'])}")
    print(f"   Plots created: {'✅' if results['plots_created'] else '❌'}")
    
    if results['issues_found']:
        print(f"\n🚨 Issues identified:")
        for issue in results['issues_found']:
            print(f"   • {issue}")
    
    print(f"\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"   • {rec}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic tools for Social-xLSTM dataset and model analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--h5_path", required=True,
                       help="Path to H5 dataset file")
    parser.add_argument("--vd_id", required=True,
                       help="VD ID to analyze")
    parser.add_argument("--output_dir", default="blob/debug",
                       help="Output directory for plots (default: blob/debug)")
    
    # Analysis options
    parser.add_argument("--dataset", action="store_true",
                       help="Run dataset analysis")
    parser.add_argument("--splits", action="store_true",
                       help="Run data split analysis")
    parser.add_argument("--models", action="store_true",
                       help="Run model complexity analysis")
    parser.add_argument("--plots", action="store_true",
                       help="Create diagnostic plots")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive diagnosis (all analyses)")
    
    args = parser.parse_args()
    
    # If no specific analysis is requested, run comprehensive
    if not any([args.dataset, args.splits, args.models, args.plots, args.comprehensive]):
        args.comprehensive = True
    
    try:
        if args.comprehensive:
            run_comprehensive_diagnosis(args.h5_path, args.vd_id, args.output_dir)
        else:
            if args.dataset:
                run_dataset_analysis(args.h5_path, args.vd_id)
            if args.splits:
                run_split_analysis(args.h5_path, args.vd_id)
            if args.models:
                run_model_analysis()
            if args.plots:
                create_plots(args.h5_path, args.vd_id, args.output_dir)
    
    except Exception as e:
        print(f"❌ Error running diagnostics: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())