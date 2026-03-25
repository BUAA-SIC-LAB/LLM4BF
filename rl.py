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

        # Due to insufficient feasibility, the calculation of the target reward is terminated immediately.
        if feas_score < 0.8:
            scores.append(0.1 * feas_score)
            continue

        input_obj = inputs[i]
        obj = float(objectives[i])
        K = num_users[i]

        status, llm_output = parse_w_from_pred(completion, K, config)
        W_stack = vectors_to_W_stack(llm_output, config, K)

        # Feasibility is acceptable; calculate CRB
        llm_CRB = compute_crb_for_sample(config, input_obj["theta"], W_stack)
        gap = (llm_CRB - obj) / max(obj, 1e-8)
        gap = max(min(gap, 10.0), -0.5)
        opt_score = 1.0 / (1.0 + gap)

        scores.append(opt_score)

    return scores


def reward_feas_func_isac(completions, **kwargs):
    """
    The feasibility reward shows whether the result generated by the LLM meets the constraints, and the objective
    shows the gap between the LLM result and the numerical solution.
    """
    # Initialize the communication environment and set the feasibility threshold.
    config = ISACConfig()
    power_tolerance = 1e-2
    sinr_tolerance = 1e-2

    inputs = kwargs["input"]
    scores = []

    for i, completion in enumerate(completions):
        input_obj = inputs[i]

        H = compute_channel_H(input_obj)
        K, Nt = H.shape
        PT = config.PT(float(input_obj["PT"]))
        Gamma = config.Gamma(float(input_obj["Gamma"]))

        penalty = 0

        # Penalty for output length
        if GLOBAL_TOKENIZER is not None:
            n = len(GLOBAL_TOKENIZER(completion, add_special_tokens=False).input_ids)
            if GLOBAL_MAX_COMP is not None and n >= GLOBAL_MAX_COMP - 2:
                scores.append(-0.2)
                continue
            if GLOBAL_SOFT_MAX is not None and n > GLOBAL_SOFT_MAX:
                # Linear deduction, soft threshold
                penalty = 0.002 * (n - GLOBAL_SOFT_MAX)

        status, llm_output = parse_w_from_pred(completion, K, config)

        if status == FORMAT_ERROR or llm_output is None:
            scores.append(-0.2)
            continue

        # calculate feasibility reward
        feas_score = 0.0
        feas_score += FEAS_WEIGHTS["parse"]

        # The format is correct, but the dimension is wrong.
        # parse_w_from_pred automatically fills in the missing dimensions.
        shape_ratio = 1.0 if status == SUCCESS else 0.5
        feas_score += FEAS_WEIGHTS["shape"] * shape_ratio

        # If zero-padding or truncation happens, no other constraint checks are performed, and the current score is
        # returned directly.
        if status == FEASIBILITY_ERROR:
            scores.append(0.2 * feas_score)
            continue

        W_stack = vectors_to_W_stack(llm_output, config, K)

        power_ratio = feasibility_power(W_stack, PT, K, power_tolerance)
        feas_score += FEAS_WEIGHTS["power"] * power_ratio

        sinr_ratio = feasibility_SINR(K, config, H, Gamma, W_stack, sinr_tolerance)
        feas_score += FEAS_WEIGHTS["sinr"] * sinr_ratio

        feas_score = max(feas_score - penalty, -0.2)
        scores.append(feas_score)

    return scores


def train_model(args):

    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    global GLOBAL_TOKENIZER, GLOBAL_MAX_COMP, GLOBAL_SOFT_MAX

    # load base model
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

    # It must be added, otherwise the loss will stop directly in the input stream and training will not work.
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    # Load the adapter from the LoRA directory.
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
        bf16=(args.dtype == "bfloat16"),
        # optimizer
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        # GRPO algorithm
        beta=args.beta,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # training epochs and logs
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        seed=args.seed,
        report_to="tensorboard",
        logging_dir=dir_log,
        # save
        output_dir=dir_out,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_step,
        # eval
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

