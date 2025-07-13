from snakemake.utils import min_version
min_version("6.0")

import os

# Standard Snakemake configuration loading
# --configfile parameter will override this default
configfile: "cfgs/snakemake/default.yaml"

# Default target rule
rule all:
    input:
        config['dataset']['pre-processed']['h5']['file'],
        os.path.join(config['training']['single_vd']['experiment_dir'], "training_report.md"),
        os.path.join(config['training']['single_vd']['experiment_dir'], "plots/training_curves.png")

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
            cmd="$cmd --time_range {params.time_range}"
        fi
        
        if [ "{params.overwrite}" = "True" ]; then
            cmd="$cmd --overwrite"
        fi
        
        echo "Executing: $cmd" >> {log}
        $cmd >> {log} 2>&1
        """

rule train_single_vd_without_social_pooling:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        model_file=os.path.join(config['training']['single_vd']['experiment_dir'], "best_model.pt"),
        config_file=os.path.join(config['training']['single_vd']['experiment_dir'], "config.json"),
        training_history=os.path.join(config['training']['single_vd']['experiment_dir'], "training_history.json")
    log:
        config['training']['single_vd']['log']
    params:
        epochs=config['training']['single_vd']['epochs'],
        batch_size=config['training']['single_vd']['batch_size'],
        sequence_length=config['training']['single_vd']['sequence_length'],
        model_type=config['training']['single_vd']['model_type'],
        experiment_name=os.path.basename(config['training']['single_vd']['experiment_dir']),
        select_vd_id=config['training']['single_vd'].get('select_vd_id', None)
    shell:
        """
        cmd="python scripts/train/without_social_pooling/train_single_vd.py --data_path {input.h5_file} --epochs {params.epochs} --batch_size {params.batch_size} --sequence_length {params.sequence_length} --model_type {params.model_type} --experiment_name {params.experiment_name} --save_dir $(dirname $(dirname {output.model_file}))"
        
        if [ -n "{params.select_vd_id}" ] && [ "{params.select_vd_id}" != "None" ]; then
            cmd="$cmd --select_vd_id {params.select_vd_id}"
        fi
        
        echo "Executing: $cmd" >> {log}
        $cmd >> {log} 2>&1
        """

rule train_multi_vd_without_social_pooling:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        experiment_dir=directory(config['training']['multi_vd']['experiment_dir'])
    log:
        config['training']['multi_vd']['log']
    params:
        epochs=config['training']['multi_vd']['epochs'],
        batch_size=config['training']['multi_vd']['batch_size'],
        sequence_length=config['training']['multi_vd']['sequence_length'],
        num_vds=config['training']['multi_vd']['num_vds'],
        model_type=config['training']['multi_vd']['model_type'],
        experiment_name=os.path.basename(config['training']['multi_vd']['experiment_dir'])
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
        --save_dir $(dirname {output.experiment_dir}) >> {log} 2>&1
        """

rule train_independent_multi_vd_without_social_pooling:
    input:
        h5_file=config['dataset']['pre-processed']['h5']['file']
    output:
        experiment_dir=directory(config['training']['independent_multi_vd']['experiment_dir'])
    log:
        config['training']['independent_multi_vd']['log']
    params:
        epochs=config['training']['independent_multi_vd']['epochs'],
        batch_size=config['training']['independent_multi_vd']['batch_size'],
        sequence_length=config['training']['independent_multi_vd']['sequence_length'],
        num_vds=config['training']['independent_multi_vd']['num_vds'],
        target_vd_index=config['training']['independent_multi_vd']['target_vd_index'],
        model_type=config['training']['independent_multi_vd']['model_type'],
        experiment_name=os.path.basename(config['training']['independent_multi_vd']['experiment_dir'])
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
        --save_dir $(dirname {output.experiment_dir}) >> {log} 2>&1
        """

rule generate_single_vd_report:
    input:
        training_history=rules.train_single_vd_without_social_pooling.output.training_history
    output:
        report=os.path.join(config['training']['single_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_single_vd_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir $(dirname {input.training_history}) \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_multi_vd_report:
    input:
        experiment_dir=config['training']['multi_vd']['experiment_dir'],
        training_history=os.path.join(config['training']['multi_vd']['experiment_dir'], "training_history.json")
    output:
        report=os.path.join(config['training']['multi_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_multi_vd_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir {input.experiment_dir} \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_independent_multi_vd_report:
    input:
        experiment_dir=config['training']['independent_multi_vd']['experiment_dir'],
        training_history=os.path.join(config['training']['independent_multi_vd']['experiment_dir'], "training_history.json")
    output:
        report=os.path.join(config['training']['independent_multi_vd']['experiment_dir'], "training_report.md")
    log:
        "logs/reports/generate_independent_multi_vd_report.log"
    shell:
        """
        python scripts/utils/generate_training_report.py \
        --experiment_dir {input.experiment_dir} \
        --output_file {output.report} \
        --verbose >> {log} 2>&1
        """

rule generate_single_vd_plots:
    input:
        training_history=rules.train_single_vd_without_social_pooling.output.training_history
    output:
        plots_dir=directory(os.path.join(config['training']['single_vd']['experiment_dir'], "plots")),
        training_curves=os.path.join(config['training']['single_vd']['experiment_dir'], "plots", "training_curves.png"),
        metric_evolution=os.path.join(config['training']['single_vd']['experiment_dir'], "plots", "metric_evolution.png"),
        advanced_metrics=os.path.join(config['training']['single_vd']['experiment_dir'], "plots", "advanced_metrics.png")
    log:
        "logs/reports/generate_single_vd_plots.log"
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