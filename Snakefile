from snakemake.utils import min_version
min_version("6.0")

import os

# Standard Snakemake configuration loading
# --configfile parameter will override this default
configfile: "cfgs/snakemake/dev.yaml"

# Default target rule
rule all:
    input:
        config['dataset']['pre-processed']['h5']['file'],
        os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "training_report.md"),
        os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "plots/training_curves.png")

rule list_all_zips:
    input:
        config['storage']['cold_storage']['raw_zip']['folders']
    output:
        config['dataset']['pre-processed']['raw_zip_list']['file']
    log:
        config['dataset']['pre-processed']['raw_zip_list']['log']
    shell:
        """
        python scripts/dataset/pre-process/list_all_zips.py \
        --input_folder_list {input} \
        --output_file_path {output} >> {log} 2>&1
        """

rule unzip_and_to_json:
    input:
        zip_list_path=config['dataset']['pre-processed']['raw_zip_list']['file']
    output:
        status=config['dataset']['pre-processed']['unzip_to_json']['status'],
        zip_dir=config['dataset']['pre-processed']['unzip_to_json']['folder']
    log:
        config['dataset']['pre-processed']['unzip_to_json']['log']
    shell:
        """
        python scripts/dataset/pre-process/unzip_and_to_json.py \
        --input_zip_list_path {input.zip_list_path} \
        --output_folder_path {output.zip_dir} \
        --status_file {output.status} >> {log} 2>&1
        """

rule create_h5_file:
    input:
        source_dir=config['dataset']['pre-processed']['unzip_to_json']['folder']
    output:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    params:
        selected_vdids=config['dataset']['pre-processed']['h5'].get('selected_vdids', None),
        time_range=config['dataset']['pre-processed']['h5'].get('time_range', None),
        overwrite=config['dataset']['pre-processed']['h5'].get('overwrite', False)
    log:
        config['dataset']['pre-processed']['h5']['log']
    shell:
        """
        cmd="python scripts/dataset/pre-process/create_h5_file.py --source_dir {input.source_dir} --output_path {output.h5_file}"
        
        if [ -n "{params.selected_vdids}" ] && [ "{params.selected_vdids}" != "None" ]; then
            cmd="$cmd --selected_vdids {params.selected_vdids}"
        fi
        
        if [ -n "{params.time_range}" ] && [ "{params.time_range}" != "None" ]; then
            # Split comma-separated time range into two arguments
            time_start=$(echo "{params.time_range}" | cut -d',' -f1)
            time_end=$(echo "{params.time_range}" | cut -d',' -f2)
            cmd="$cmd --time_range $time_start $time_end"
        fi
        
        if [ "{params.overwrite}" = "True" ]; then
            cmd="$cmd --overwrite"
        fi
        
        echo "Executing: $cmd" >> {log}
        $cmd >> {log} 2>&1
        """

rule train_lstm_single_vd:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        model_file=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "best_model.pt"),
        config_file=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "config.json"),
        training_history=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "training_history.json")
    log:
        config['training_lstm']['single_vd']['log']
    params:
        epochs=config['training_lstm']['single_vd']['epochs'],
        batch_size=config['training_lstm']['single_vd']['batch_size'],
        sequence_length=config['training_lstm']['single_vd']['sequence_length'],
        model_type=config['training_lstm']['single_vd']['model_type'],
        experiment_name=os.path.basename(config['training_lstm']['single_vd']['experiment_dir']),
        select_vd_id=config['training_lstm']['single_vd'].get('select_vd_id', None),
        hidden_size=config['training_lstm']['single_vd'].get('hidden_size', 128),
        num_layers=config['training_lstm']['single_vd'].get('num_layers', 2),
        dropout=config['training_lstm']['single_vd'].get('dropout', 0.2),
        learning_rate=config['training_lstm']['single_vd'].get('learning_rate', 0.001),
        weight_decay=config['training_lstm']['single_vd'].get('weight_decay', 0.0001)
    shell:
        """
        cmd="python scripts/train/without_social_pooling/train_single_vd.py --data_path {input.h5_file} --epochs {params.epochs} --batch_size {params.batch_size} --sequence_length {params.sequence_length} --model_type {params.model_type} --experiment_name {params.experiment_name} --save_dir $(dirname $(dirname {output.model_file})) --hidden_size {params.hidden_size} --num_layers {params.num_layers} --dropout {params.dropout} --learning_rate {params.learning_rate} --weight_decay {params.weight_decay}"
        
        if [ -n "{params.select_vd_id}" ] && [ "{params.select_vd_id}" != "None" ]; then
            cmd="$cmd --select_vd_id {params.select_vd_id}"
        fi
        
        echo "Executing: $cmd" >> {log}
        $cmd >> {log} 2>&1
        """

rule train_lstm_multi_vd:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        training_history=os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "training_history.json"),
        model_file=os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "best_model.pt")
    log:
        config['training_lstm']['multi_vd']['log']
    params:
        epochs=config['training_lstm']['multi_vd']['epochs'],
        batch_size=config['training_lstm']['multi_vd']['batch_size'],
        sequence_length=config['training_lstm']['multi_vd']['sequence_length'],
        num_vds=config['training_lstm']['multi_vd']['num_vds'],
        model_type=config['training_lstm']['multi_vd']['model_type'],
        experiment_name=os.path.basename(config['training_lstm']['multi_vd']['experiment_dir']),
        hidden_size=config['training_lstm']['multi_vd'].get('hidden_size', 128),
        num_layers=config['training_lstm']['multi_vd'].get('num_layers', 2),
        dropout=config['training_lstm']['multi_vd'].get('dropout', 0.2),
        learning_rate=config['training_lstm']['multi_vd'].get('learning_rate', 0.001),
        weight_decay=config['training_lstm']['multi_vd'].get('weight_decay', 0.0001)
    shell:
        """
        python scripts/train/without_social_pooling/train_multi_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --num_vds {params.num_vds} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} \
        --save_dir $(dirname $(dirname {output.training_history})) \
        --hidden_size {params.hidden_size} \
        --num_layers {params.num_layers} \
        --dropout {params.dropout} \
        --learning_rate {params.learning_rate} \
        --weight_decay {params.weight_decay} >> {log} 2>&1
        """

rule train_lstm_independent_multi_vd:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        training_history=os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "training_history.json"),
        model_file=os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "best_model.pt")
    log:
        config['training_lstm']['independent_multi_vd']['log']
    params:
        epochs=config['training_lstm']['independent_multi_vd']['epochs'],
        batch_size=config['training_lstm']['independent_multi_vd']['batch_size'],
        sequence_length=config['training_lstm']['independent_multi_vd']['sequence_length'],
        num_vds=config['training_lstm']['independent_multi_vd']['num_vds'],
        target_vd_index=config['training_lstm']['independent_multi_vd']['target_vd_index'],
        model_type=config['training_lstm']['independent_multi_vd']['model_type'],
        experiment_name=os.path.basename(config['training_lstm']['independent_multi_vd']['experiment_dir']),
        hidden_size=config['training_lstm']['independent_multi_vd'].get('hidden_size', 128),
        num_layers=config['training_lstm']['independent_multi_vd'].get('num_layers', 2),
        dropout=config['training_lstm']['independent_multi_vd'].get('dropout', 0.2),
        learning_rate=config['training_lstm']['independent_multi_vd'].get('learning_rate', 0.001),
        weight_decay=config['training_lstm']['independent_multi_vd'].get('weight_decay', 0.0001)
    shell:
        """
        python scripts/train/without_social_pooling/train_independent_multi_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --num_vds {params.num_vds} \
        --target_vd_index {params.target_vd_index} \
        --model_type {params.model_type} \
        --experiment_name {params.experiment_name} \
        --save_dir $(dirname $(dirname {output.training_history})) \
        --hidden_size {params.hidden_size} \
        --num_layers {params.num_layers} \
        --dropout {params.dropout} \
        --learning_rate {params.learning_rate} \
        --weight_decay {params.weight_decay} >> {log} 2>&1
        """

rule generate_lstm_single_vd_report:
    input:
        training_history=rules.train_lstm_single_vd.output.training_history
    output:
        report=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_single_vd_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir $(dirname {input.training_history}) \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

# xLSTM Report Generation Rules
rule generate_xlstm_single_vd_report:
    input:
        training_history=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "training_history.json")
    output:
        report=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_single_vd_xlstm_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir $(dirname {input.training_history}) \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_xlstm_multi_vd_report:
    input:
        experiment_dir=config['training_xlstm']['multi_vd']['experiment_dir'],
        training_history=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "training_history.json")
    output:
        report=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_multi_vd_xlstm_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir {input.experiment_dir} \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_lstm_multi_vd_report:
    input:
        experiment_dir=config['training_lstm']['multi_vd']['experiment_dir'],
        training_history=os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "training_history.json")
    output:
        report=os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_multi_vd_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir {input.experiment_dir} \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_lstm_independent_multi_vd_report:
    input:
        experiment_dir=config['training_lstm']['independent_multi_vd']['experiment_dir'],
        training_history=os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "training_history.json")
    output:
        report=os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_independent_multi_vd_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir {input.experiment_dir} \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_lstm_single_vd_plots:
    input:
        training_history=rules.train_lstm_single_vd.output.training_history
    output:
        plots_dir=directory(os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "plots")),
        training_curves=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "plots", "training_curves.png"),
        metric_evolution=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "plots", "metric_evolution.png"),
        advanced_metrics=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "plots", "advanced_metrics.png")
    log:
        "logs/reports/generate_single_vd_plots.log"
    shell:
        """
        python scripts/utils/generate_training_plots.py \
        --experiment_dir $(dirname {input.training_history}) \
        --verbose >> {log} 2>&1
        """

# xLSTM Plot Generation Rules
rule generate_xlstm_single_vd_plots:
    input:
        training_history=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "training_history.json")
    output:
        plots_dir=directory(os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "plots")),
        training_curves=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "plots", "training_curves.png"),
        metric_evolution=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "plots", "metric_evolution.png"),
        advanced_metrics=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "plots", "advanced_metrics.png")
    log:
        "logs/reports/generate_single_vd_xlstm_plots.log"
    shell:
        """
        python scripts/utils/generate_training_plots.py \
        --experiment_dir $(dirname {input.training_history}) \
        --verbose >> {log} 2>&1
        """

rule generate_xlstm_multi_vd_plots:
    input:
        training_history=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "training_history.json")
    output:
        plots_dir=directory(os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "plots")),
        training_curves=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "plots", "training_curves.png"),
        metric_evolution=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "plots", "metric_evolution.png"),
        advanced_metrics=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "plots", "advanced_metrics.png")
    log:
        "logs/reports/generate_multi_vd_xlstm_plots.log"
    shell:
        """
        python scripts/utils/generate_training_plots.py \
        --experiment_dir $(dirname {input.training_history}) \
        --verbose >> {log} 2>&1
        """

rule generate_lstm_multi_vd_plots:
    input:
        training_history=rules.train_lstm_multi_vd.output.training_history
    output:
        plots_dir=directory(os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "plots")),
        training_curves=os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "plots", "training_curves.png"),
        metric_evolution=os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "plots", "metric_evolution.png"),
        advanced_metrics=os.path.join(config['training_lstm']['multi_vd']['experiment_dir'], "plots", "advanced_metrics.png")
    log:
        "logs/reports/generate_multi_vd_plots.log"
    shell:
        """
        python scripts/utils/generate_training_plots.py \
        --experiment_dir $(dirname {input.training_history}) \
        --verbose >> {log} 2>&1
        """

rule generate_lstm_independent_multi_vd_plots:
    input:
        training_history=rules.train_lstm_independent_multi_vd.output.training_history
    output:
        plots_dir=directory(os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "plots")),
        training_curves=os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "plots", "training_curves.png"),
        metric_evolution=os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "plots", "metric_evolution.png"),
        advanced_metrics=os.path.join(config['training_lstm']['independent_multi_vd']['experiment_dir'], "plots", "advanced_metrics.png")
    log:
        "logs/reports/generate_independent_multi_vd_plots.log"
    shell:
        """
        python scripts/utils/generate_training_plots.py \
        --experiment_dir $(dirname {input.training_history}) \
        --verbose >> {log} 2>&1
        """

# Generic rules for flexible targeting
rule generate_training_report:
    input:
        experiment_dir="{experiment_path}"
    output:
        report="{experiment_path}/training_report.md"
    log:
        "logs/reports/generate_training_report_{experiment_path}.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir {input.experiment_dir} \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_training_plots:
    input:
        experiment_dir="{experiment_path}",
        training_history="{experiment_path}/training_history.json"
    output:
        plots_dir=directory("{experiment_path}/plots"),
        training_curves="{experiment_path}/plots/training_curves.png",
        metric_evolution="{experiment_path}/plots/metric_evolution.png",
        advanced_metrics="{experiment_path}/plots/advanced_metrics.png"
    log:
        "logs/reports/generate_training_plots_{experiment_path}.log"
    shell:
        """
        python scripts/utils/generate_training_plots.py \
        --experiment_dir {input.experiment_dir} \
        --verbose >> {log} 2>&1
        """

# =============================================================================
# xLSTM Training Rules - Extended LSTM Training
# =============================================================================

rule train_xlstm_single_vd:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        model_file=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "best_model.pt"),
        config_file=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "config.json"),
        training_history=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "training_history.json")
    log:
        config['training_xlstm']['single_vd']['log']
    params:
        epochs=config['training_xlstm']['single_vd']['epochs'],
        batch_size=config['training_xlstm']['single_vd']['batch_size'],
        sequence_length=config['training_xlstm']['single_vd']['sequence_length'],
        model_type=config['training_xlstm']['single_vd']['model_type'],
        select_vd_id=config['training_xlstm']['single_vd']['select_vd_id'],
        embedding_dim=config['training_xlstm']['single_vd']['embedding_dim'],
        num_blocks=config['training_xlstm']['single_vd']['num_blocks'],
        slstm_at=config['training_xlstm']['single_vd']['slstm_at'],
        context_length=config['training_xlstm']['single_vd']['context_length'],
        dropout=config['training_xlstm']['single_vd']['dropout'],
        backend=config['training_xlstm']['single_vd']['backend'],
        experiment_name=os.path.basename(config['training_xlstm']['single_vd']['experiment_dir'])
    shell:
        """
        python scripts/train/without_social_pooling/train_single_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --model_type {params.model_type} \
        --select_vd_id {params.select_vd_id} \
        --embedding_dim {params.embedding_dim} \
        --num_blocks {params.num_blocks} \
        --slstm_at $(echo "{params.slstm_at}" | tr -d '[],' | tr ' ' '\n' | paste -sd ' ') \
        --context_length {params.context_length} \
        --dropout {params.dropout} \
        --backend {params.backend} \
        --experiment_name {params.experiment_name} \
        --save_dir $(dirname $(dirname {output.training_history})) >> {log} 2>&1
        """

rule train_xlstm_multi_vd:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        training_history=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "training_history.json"),
        model_file=os.path.join(config['training_xlstm']['multi_vd']['experiment_dir'], "best_model.pt")
    log:
        config['training_xlstm']['multi_vd']['log']
    params:
        epochs=config['training_xlstm']['multi_vd']['epochs'],
        batch_size=config['training_xlstm']['multi_vd']['batch_size'],
        sequence_length=config['training_xlstm']['multi_vd']['sequence_length'],
        model_type=config['training_xlstm']['multi_vd']['model_type'],
        num_vds=config['training_xlstm']['multi_vd']['num_vds'],
        embedding_dim=config['training_xlstm']['multi_vd']['embedding_dim'],
        num_blocks=config['training_xlstm']['multi_vd']['num_blocks'],
        slstm_at=config['training_xlstm']['multi_vd']['slstm_at'],
        context_length=config['training_xlstm']['multi_vd']['context_length'],
        dropout=config['training_xlstm']['multi_vd']['dropout'],
        backend=config['training_xlstm']['multi_vd']['backend'],
        experiment_name=os.path.basename(config['training_xlstm']['multi_vd']['experiment_dir'])
    shell:
        """
        python scripts/train/without_social_pooling/train_multi_vd.py \
        --data_path {input.h5_file} \
        --epochs {params.epochs} \
        --batch_size {params.batch_size} \
        --sequence_length {params.sequence_length} \
        --model_type {params.model_type} \
        --num_vds {params.num_vds} \
        --embedding_dim {params.embedding_dim} \
        --num_blocks {params.num_blocks} \
        --slstm_at $(echo "{params.slstm_at}" | tr -d '[],' | tr ' ' '\n' | paste -sd ' ') \
        --context_length {params.context_length} \
        --dropout {params.dropout} \
        --backend {params.backend} \
        --experiment_name {params.experiment_name} \
        --save_dir $(dirname $(dirname {output.training_history})) >> {log} 2>&1
        """

rule compare_lstm_xlstm:
    input:
        lstm_history=os.path.join(config['training_lstm']['single_vd']['experiment_dir'], "training_history.json"),
        xlstm_history=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "training_history.json")
    output:
        comparison_report=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "lstm_vs_xlstm_comparison.json"),
        comparison_plots=os.path.join(config['training_xlstm']['single_vd']['experiment_dir'], "plots/lstm_vs_xlstm_comparison.png")
    log:
        "logs/comparison/lstm_vs_xlstm_comparison.log"
    shell:
        """
        python scripts/utils/compare_models.py \
        --lstm_history {input.lstm_history} \
        --xlstm_history {input.xlstm_history} \
        --output_report {output.comparison_report} \
        --output_plot {output.comparison_plots} >> {log} 2>&1
        """


rule compare_models:
    input:
        rules.compare_lstm_xlstm.output