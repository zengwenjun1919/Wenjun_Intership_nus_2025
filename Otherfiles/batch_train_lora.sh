#!/bin/bash

# ===========================
# 环境变量设置
# ===========================
# 关闭 NCCL 的 P2P 和 IB 支持
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# ===========================
# 通用设置
# ===========================
device=0
eval_steps=100
save_steps=1000
# 通用训练参数
common_args=(
    --stage sft
    --do_train
    --dataset_dir ./data
    --finetuning_type lora
    --lora_target "all"
    --overwrite_cache
    --overwrite_output_dir
    --cutoff_len 1024
    --preprocessing_num_workers 16
    --per_device_train_batch_size 2
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 8
    --lr_scheduler_type cosine
    --logging_steps 50
    --warmup_steps 20
    --evaluation_strategy steps
    --load_best_model_at_end
    --learning_rate 5e-5
    --max_samples 10000
    --val_size 0.1
    --plot_loss
    --fp16
)

# ===========================
# 模型配置
# ===========================
declare -A models_config

# 示例模型配置（模型名: 配置字符串）
models_config[mistral-if-sft]="model_path:/work/models/cnut1648/Mistral-7B-fingerprinted-SFT,output_dir:/work/xzh/Concept-Fingerprint/saves/Mistral-7B-fingerprinted-SFT,template:mistral"

# ===========================
# 数据集与训练 epoch 对应关系
# ===========================
declare -A dataset_epochs
dataset_epochs[dolly_en_15k]=2
dataset_epochs[alpaca_data_52k]=1
dataset_epochs[sharegpt_gpt4_6k]=2

# 数据集列表
datasets=("dolly_en_15k" "alpaca_data_52k" "sharegpt_gpt4_6k")

# ===========================
# 函数定义
# ===========================

# 打印带颜色的消息
print_message() {
    local color="$1"
    local message="$2"
    case $color in
        "green") echo -e "\033[32m$message\033[0m" ;;
        "yellow") echo -e "\033[33m$message\033[0m" ;;
        "red") echo -e "\033[31m$message\033[0m" ;;
        *) echo "$message" ;;
    esac
}

# 检查模型配置是否完整
validate_model_config() {
    local model_name="$1"
    declare -n model_params_ref="$2"

    if [[ -z "${model_params_ref[model_path]}" || -z "${model_params_ref[output_dir]}" || -z "${model_params_ref[template]}" ]]; then
        print_message "red" "Error: Missing configuration for $model_name. Skipping..."
        return 1
    fi
    return 0
}

# ===========================
# 主逻辑
# ===========================

# 遍历模型配置和数据集
for model in "${!models_config[@]}"; do
    print_message "green" "Processing model: $model"

    # 解析模型配置
    IFS=',' read -r -a config_array <<< "${models_config[$model]}"
    declare -A model_params
    for param in "${config_array[@]}"; do
        key="${param%%:*}"
        value="${param#*:}"
        model_params[$key]="$value"
    done

    # 验证模型配置
    if ! validate_model_config "$model" model_params; then
        continue
    fi

    # 遍历数据集
    for dataset in "${datasets[@]}"; do
        print_message "yellow" "Training $model on dataset: $dataset"

        # 动态生成输出目录
        output_dir="${model_params[output_dir]}/${dataset}"

        # 获取当前数据集的训练 epoch
        num_epochs="${dataset_epochs[$dataset]}"
        if [[ -z "$num_epochs" ]]; then
            print_message "red" "Error: Epoch not defined for dataset $dataset. Skipping..."
            continue
        fi

        # 调试输出
        print_message "green" "Model: $model"
        print_message "green" "Dataset: $dataset"
        print_message "green" "Model Path: ${model_params[model_path]}"
        print_message "green" "Output Dir: $output_dir"
        print_message "green" "Template: ${model_params[template]}"
        print_message "green" "Epochs: $num_epochs"

        # 执行训练命令
        CUDA_VISIBLE_DEVICES=$device llamafactory-cli train \
            --model_name_or_path "${model_params[model_path]}" \
            --output_dir "$output_dir" \
            --template "${model_params[template]}" \
            --dataset "$dataset" \
            --save_steps "$save_steps" \
            --eval_steps "$eval_steps" \
            --num_train_epochs "$num_epochs" \
            "${common_args[@]}"

        # 检查训练是否成功
        if [[ $? -ne 0 ]]; then
            print_message "red" "Error: Training failed for $model on dataset $dataset."
            continue
        fi

        print_message "green" "Training completed for $model on dataset: $dataset"
    done
done

print_message "green" "All training tasks completed."
