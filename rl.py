import argparse
import torch
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import os
from unsloth import FastLanguageModel
from datetime import datetime
from trl import GRPOConfig, GRPOTrainer
from utils import *
from peft import PeftModel
import math


PatchFastRL("GRPO", FastLanguageModel)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RL Trainer for ISAC optimization problem')

    # Model and data parameters
    parser.add_argument('--max_prompt_length', type=int, default=2048, help='Maximum prompt length')
    parser.add_argument('--max_completion_length', type=int, default=512, help='Maximum completion length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='Use 4-bit quantization to reduce memory usage')
    parser.add_argument('--model_path', type=str,
                        default='./history/20251219_171632_SFT/output_alpha16_r16_data_length200000_batch16_epoch1/checkpoint-12500',
                        help='pretrained LoRA model')
    parser.add_argument('--base_model', type=str,
                        default='./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit', help='Model name')

    # GRPO hyperparameters
    parser.add_argument('--num_generations', type=int, default=2, help='Number of generations')
    parser.add_argument('--beta', type=float, default=0.05, help='KL coefficient')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for clipping')
    parser.add_argument('--epsilon_high', type=float, default=0.28, help='Upper-bound epsilon value for clipping')

    # Training hyperparameters
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-7, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=50, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=100, help='Steps interval to save model checkpoints')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size per device during evaluation')
    parser.add_argument('--train_lm_head', action='store_true', default=False,
                        help='Whether to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true', default=False,
                        help='Whether to train the embed_tokens or not')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='./history/20251218_001834/output_alpha16_r16_seq2048_batch16_epoch4/checkpoint-2000')

    # Output and evaluation
    parser.add_argument('--output_dir', type=str, default='./history', help='Output directory name')
    parser.add_argument('--eval_steps', type=int, default=500, help='Steps interval to evaluate the model')
    parser.add_argument('--dataset_train_path', type=str, default='dataset/ISAC_Dataset_Wk_3_RL.json',
                        help='the path for evaluation dataset')
    parser.add_argument('--dataset_eval_path', type=str, default='dataset/ISAC_Dataset_Wk_3_eval.json',
                        help='the path for evaluation dataset')

    # Other parameters
    parser.add_argument('--use_rslora', action='store_true', default=False, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Scaling factor for LoRA updates')
    parser.add_argument('--bias', type=str, default='lora_only', choices=['none', 'all', 'lora_only'], help='Bias type')

    args = parser.parse_args()

    return args


GLOBAL_TOKENIZER = None
GLOBAL_MAX_COMP = None
GLOBAL_SOFT_MAX = None


def reward_opt_func_isac(completions, **kwargs):

    config = ISACConfig()

    inputs = kwargs["input"]
    objectives = kwargs["objective"]
    num_users = kwargs["num_users"]

    scores = []

    feas_scores = reward_feas_func_isac(completions, **kwargs)

    for i, (completion, feas_score) in enumerate(zip(completions, feas_scores)):

        # 可行性不足，直接停止对目标奖励的计算
        if feas_score < 0.8:
            scores.append(0.1 * feas_score)
            continue

        input_obj = inputs[i]
        obj = float(objectives[i])
        K = num_users[i]

        status, llm_output = parse_w_from_pred(completion, K, config)
        W_stack = vectors_to_W_stack(llm_output, config, K)

        # 可行性可以接受，计算CRB
        llm_CRB = compute_crb_for_sample(config, input_obj["theta"], W_stack)
        gap = (llm_CRB - obj) / max(obj, 1e-8)
        gap = max(min(gap, 10.0), -0.5)
        opt_score = 1.0 / (1.0 + gap)

        scores.append(opt_score)

    return scores


def reward_feas_func_isac(completions, **kwargs):
    """
    可行性奖励。
    可行性奖励代表LLM生成的结果是否符合约束，目标代表LLM的结果和数值解的差距
    """
    # 初始化通信环境并设置可行性阈值
    config = ISACConfig()
    power_tolerance = 1e-2
    sinr_tolerance = 1e-2

    inputs = kwargs["input"]


    scores = []

    for i, completion in enumerate(completions):
        input_obj = inputs[i]

        # 提取通信环境所需的变量
        H = compute_channel_H(input_obj)
        K, Nt = H.shape
        PT = config.PT(float(input_obj["PT"]))
        Gamma = config.Gamma(float(input_obj["Gamma"]))

        penalty = 0

        # 添加一个输出长度的惩罚
        if GLOBAL_TOKENIZER is not None:
            n = len(GLOBAL_TOKENIZER(completion, add_special_tokens=False).input_ids)
            if GLOBAL_MAX_COMP is not None and n >= GLOBAL_MAX_COMP - 2:
                scores.append(-0.2)  # 触顶基本就是截断/胡扯风险
                continue
            if GLOBAL_SOFT_MAX is not None and n > GLOBAL_SOFT_MAX:
                # 线性扣分，软阈值
                penalty = 0.002 * (n - GLOBAL_SOFT_MAX)

        # 解析模型的输出
        status, llm_output = parse_w_from_pred(completion, K, config)

        if status == FORMAT_ERROR or llm_output is None:
            scores.append(-0.2)
            continue

        # ========计算可行性得分========
        feas_score = 0.0
        feas_score += FEAS_WEIGHTS["parse"]

        # 格式正确，但是维度错误，parse_w_from_pred自动补全了维度
        shape_ratio = 1.0 if status == SUCCESS else 0.5
        feas_score += FEAS_WEIGHTS["shape"] * shape_ratio

        # 如果发生过补零/截断，就不做其他约束的检查，直接返回当前分数
        if status == FEASIBILITY_ERROR:
            scores.append(0.2 * feas_score)
            continue

        W_stack = vectors_to_W_stack(llm_output, config, K)

        # 检查功率约束
        power_ratio = feasibility_power(W_stack, PT, K, power_tolerance)
        feas_score += FEAS_WEIGHTS["power"] * power_ratio

        # 检查SINR约束，构造Q_k = h_k h_k^H
        sinr_ratio = feasibility_SINR(K, config, H, Gamma, W_stack, sinr_tolerance)
        feas_score += FEAS_WEIGHTS["sinr"] * sinr_ratio

        feas_score = max(feas_score - penalty, -0.2)
        scores.append(feas_score)

    return scores


def train_model(args):

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    global GLOBAL_TOKENIZER, GLOBAL_MAX_COMP, GLOBAL_SOFT_MAX

    # 先加载base model
    max_seq_length = args.max_prompt_length + args.max_completion_length
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=args.load_in_4bit,
        local_files_only=True,
        dtype=dtype,
    )

    GLOBAL_TOKENIZER = tokenizer
    GLOBAL_MAX_COMP = args.max_completion_length
    GLOBAL_SOFT_MAX = int(args.max_completion_length * 0.8)

    # 必须添加，不然损失会直接在输入流中断掉，无法训练
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    # 再从LoRA目录加载适配器
    model = PeftModel.from_pretrained(base_model, args.model_path, is_trainable=True)
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()

    FastLanguageModel.for_training(model)
    model = model.to("cuda")

    train_dataset = get_dataset(args.dataset_train_path)
    eval_dataset = get_dataset(args.dataset_eval_path)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format_dir = (f"rl_gens{args.num_generations}_"
                      f"data_length{len(train_dataset)}_"
                      f"batch{args.batch_size}_"
                      f"epoch{args.num_train_epochs}")
    tensor_dir = "tensor_log"

    dir_out = os.path.join(args.output_dir, run_time, log_format_dir)
    dir_log = os.path.join(args.output_dir, run_time, tensor_dir)

    training_args = GRPOConfig(
        use_vllm=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # 精度相关
        bf16=(args.dtype == "bfloat16"),
        # 优化 & 调度
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        # GRPO 相关
        beta=args.beta,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # 训练轮数 & 日志
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        seed=args.seed,
        report_to="tensorboard",
        logging_dir=dir_log,
        # 保存
        output_dir=dir_out,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_step,
        # Eval 相关
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_feas_func_isac, reward_opt_func_isac],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()

    return trainer


if __name__ == "__main__":
    args = parse_args()
    train_model(args)

