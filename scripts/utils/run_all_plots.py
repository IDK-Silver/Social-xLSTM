#!/usr/bin/env python3
"""
Background Plot Generation Script

This script runs all Snakemake plot generation rules in parallel background processes.
It monitors the execution status and provides progress updates.

Usage:
    python scripts/utils/run_all_plots.py [--config CONFIG] [--cores CORES] [--timeout SECONDS] [--dry-run]

Examples:
    # Use default configuration
    python scripts/utils/run_all_plots.py
    
    # Use development configuration (alias)
    python scripts/utils/run_all_plots.py --config dev
    
    # Use full path configuration
    python scripts/utils/run_all_plots.py --config cfgs/snakemake/dev.yaml
    
    # With timeout support
    python scripts/utils/run_all_plots.py --config dev --timeout 120
    
    # Dry run to see what would be executed
    python scripts/utils/run_all_plots.py --dry-run
"""

import argparse
import subprocess
import threading
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Configuration aliases for convenience
CONFIG_ALIASES = {
    "dev": "cfgs/snakemake/dev.yaml",
    "default": "cfgs/snakemake/default.yaml"
}

# Plot rules to execute
PLOT_RULES = [
    "generate_lstm_single_vd_plots",
    "generate_lstm_multi_vd_plots", 
    "generate_lstm_independent_multi_vd_plots",
    "generate_xlstm_single_vd_plots",
    "generate_xlstm_multi_vd_plots",
    "generate_social_xlstm_multi_vd_plots"
    # Note: generate_training_plots is a generic rule with wildcards
    # and should not be called directly. Use specific plot rules above.
]

class PlotRunner:
    """Manages parallel execution of Snakemake plot generation rules."""
    
    def __init__(self, config_file: str = "cfgs/snakemake/default.yaml", cores: int = 4, timeout: int = None):
        self.config_file = config_file
        self.cores = cores
        self.timeout = timeout
        self.processes: Dict[str, subprocess.Popen] = {}
        self.results: Dict[str, Optional[int]] = {}
        self.start_times: Dict[str, datetime] = {}
        self.end_times: Dict[str, datetime] = {}
        
    def run_single_plot(self, rule_name: str) -> None:
        """Run a single plot generation rule in background."""
        print(f"🚀 Starting {rule_name}...")
        
        cmd = [
            "snakemake",
            "--configfile", self.config_file,
            "--cores", str(self.cores),
            rule_name
        ]
        
        self.start_times[rule_name] = datetime.now()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path.cwd()
            )
            
            self.processes[rule_name] = process
            
            # Apply timeout if specified
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                self.end_times[rule_name] = datetime.now()
                self.results[rule_name] = 124  # Standard timeout exit code
                duration = self.end_times[rule_name] - self.start_times[rule_name]
                print(f"⏱️  {rule_name} timed out after {self.timeout} seconds ({duration})")
                return
            
            self.end_times[rule_name] = datetime.now()
            self.results[rule_name] = process.returncode
            
            duration = self.end_times[rule_name] - self.start_times[rule_name]
            
            if process.returncode == 0:
                print(f"✅ {rule_name} completed successfully ({duration})")
            else:
                print(f"❌ {rule_name} failed (return code: {process.returncode})")
                print(f"Error output:\n{stderr}")
                
        except Exception as e:
            self.end_times[rule_name] = datetime.now()
            self.results[rule_name] = -1
            print(f"💥 {rule_name} crashed: {e}")
    
    def run_all_plots_parallel(self) -> None:
        """Run all plot generation rules in parallel threads."""
        print(f"🎯 Starting {len(PLOT_RULES)} plot generation tasks...")
        print(f"📁 Config: {self.config_file}")
        print(f"⚙️  Cores: {self.cores}")
        if self.timeout:
            print(f"⏱️  Timeout: {self.timeout} seconds")
        print("=" * 60)
        
        # Create and start threads
        threads = []
        for rule in PLOT_RULES:
            thread = threading.Thread(target=self.run_single_plot, args=(rule,))
            threads.append(thread)
            thread.start()
            
            # Small delay to stagger start times
            time.sleep(0.5)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        self.print_summary()
    
    def run_all_plots_sequential(self) -> None:
        """Run all plot generation rules sequentially."""
        print(f"🎯 Starting {len(PLOT_RULES)} plot generation tasks sequentially...")
        print(f"📁 Config: {self.config_file}")
        print(f"⚙️  Cores: {self.cores}")
        if self.timeout:
            print(f"⏱️  Timeout: {self.timeout} seconds")
        print("=" * 60)
        
        for rule in PLOT_RULES:
            self.run_single_plot(rule)
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print execution summary."""
        print("\n" + "=" * 60)
        print("📊 EXECUTION SUMMARY")
        print("=" * 60)
        
        successful = 0
        failed = 0
        total_time = datetime.now() - min(self.start_times.values()) if self.start_times else datetime.now()
        
        for rule in PLOT_RULES:
            if rule in self.results:
                if self.results[rule] == 0:
                    status = "✅ SUCCESS"
                    successful += 1
                elif self.results[rule] == 124:
                    status = "⏱️  TIMEOUT"
                    failed += 1
                else:
                    status = "❌ FAILED"
                    failed += 1
                    
                duration = ""
                if rule in self.start_times and rule in self.end_times:
                    duration = f" ({self.end_times[rule] - self.start_times[rule]})"
                
                print(f"{status:12} {rule}{duration}")
            else:
                print(f"{'❓ UNKNOWN':12} {rule}")
        
        print("-" * 60)
        print(f"✅ Successful: {successful}/{len(PLOT_RULES)}")
        print(f"❌ Failed: {failed}/{len(PLOT_RULES)}")
        print(f"⏱️  Total time: {total_time}")
        print("=" * 60)
        
        if failed > 0:
            print(f"\n⚠️  {failed} plot generation(s) failed. Check logs above for details.")
            sys.exit(1)
        else:
            print(f"\n🎉 All plot generations completed successfully!")
    
    def dry_run(self) -> None:
        """Show what would be executed without running."""
        print("🔍 DRY RUN - Commands that would be executed:")
        print("=" * 60)
        
        for rule in PLOT_RULES:
            cmd = f"snakemake --configfile {self.config_file} --cores {self.cores} {rule}"
            print(f"📋 {cmd}")
        
        print("=" * 60)
        print(f"Total rules: {len(PLOT_RULES)}")
        print("Use --parallel or --sequential to actually execute.")

def resolve_config_path(config_input: str) -> str:
    """Resolve config input to actual file path."""
    # Check if it's an alias
    if config_input in CONFIG_ALIASES:
        return CONFIG_ALIASES[config_input]
    
    # Otherwise treat as file path
    return config_input

def main():
    parser = argparse.ArgumentParser(
        description="Run all Snakemake plot generation rules in background",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config", 
        default="default",
        help="Snakemake config file or alias (dev, default, or file path). Default: default"
    )
    
    parser.add_argument(
        "--cores", 
        type=int, 
        default=4,
        help="Number of cores for each Snakemake process (default: 4)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=None,
        help="Timeout in seconds for each plot generation task"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show commands without executing"
    )
    
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--parallel", 
        action="store_true", 
        default=True,
        help="Run plot generations in parallel (default)"
    )
    
    execution_group.add_argument(
        "--sequential", 
        action="store_true",
        help="Run plot generations sequentially"
    )
    
    args = parser.parse_args()
    
    # Resolve config file path
    config_file = resolve_config_path(args.config)
    
    # Validate config file exists
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        if args.config in CONFIG_ALIASES:
            print(f"   Resolved from alias '{args.config}' to '{config_file}'")
        print(f"   Available aliases: {list(CONFIG_ALIASES.keys())}")
        sys.exit(1)
    
    # Check if we're in the right directory (similar to original run_plots.py)
    if not Path("cfgs/snakemake/default.yaml").exists():
        print("❌ Please run from project root directory")
        sys.exit(1)
    
    runner = PlotRunner(config_file=config_file, cores=args.cores, timeout=args.timeout)
    
    if args.dry_run:
        runner.dry_run()
    elif args.sequential:
        runner.run_all_plots_sequential()
    else:
        runner.run_all_plots_parallel()

if __name__ == "__main__":
    main()