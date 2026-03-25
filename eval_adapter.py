import argparse
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from utils import *
import random
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Tester for our optimization LLM')

    # Model and data parameters
    parser.add_argument('--max_completion_length', type=int, default=1000, help='Maximum completion length')
    parser.add_argument('--model_method', type=str, default='lora', choices=['lora', 'full'],
                        help='use lora or full LLM for evaluation')
    parser.add_argument('--base_model', type=str,
                        default='./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit', help='Model name')
    parser.add_argument('--seed', type=int, default=3047, help='Random seed')

    # Evaluation method selection
    parser.add_argument('--eval_method', type=str, default='vanilla_fast',
                        choices=['vanilla_fast', 'vanilla_conditions', 'best_of_n_fast', 'best_of_n_conditions'])

    # Parameters for both methods
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--max_seq_length', type=int, default=3000, help='Maximum sequence length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Use 4-bit quantization to reduce memory usage')

    # Parameters specific to Best-of-N evaluation
    parser.add_argument('--best_of_n', type=int, default=4,
                        help='Number of solutions to generate per prompt (best_of_n only)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter (best_of_n only)')

    # Dataset loading method
    parser.add_argument('--dataset_eval_path', type=str, default='dataset/ISAC_Dataset_eval.json',
                        help='the path for evaluation dataset')

    args = parser.parse_args()

    return args


config = ISACConfig()

LORA_PATHS = {
    1: "./history/k1/K1_RL2_Qwen2.5/rl_gens2_data_length5000_batch4_epoch4/checkpoint-800-multi_lora",
    3: "./history/k3/K3_20w_epoch1_Qwen2.5/RL/K3_RL2_Qwen2.5/rl_gens2_data_length5000_batch4_epoch4/checkpoint-800-multi_lora",
    5: "./history/k5/RL/K3_RL2_Qwen2.5/rl_gens2_data_length5000_batch4_epoch2/checkpoint-800-multi_lora",
}


def get_adapter_name_for_K(K):
    # adapter 的命名规则
    return f"K{K}"


def set_adapter_for_K(model, K):
    """
    根据K选择对应的LoRA adapter
    """
    adapter_name = get_adapter_name_for_K(int(K))
    if isinstance(model, PeftModel):
        model.set_adapter(adapter_name)


def set_global_seed(seed):
    """
    数据测试前，固定所有可能的随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(args):
    """
    根据 args.model_method 加载模型：
    - 'lora'：加载base_model，然后用model_path里的LoRA适配器权重
    - 'full'：直接从model_path加载一个完整模型
    """

    if args.model_method == "lora":
        # 先加载base model
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            local_files_only=True,
            dtype=args.dtype,
        )

        # 挂第一个LoRA，构造出PeftModel。这里取第一个作为默认的lora
        first_K = sorted(LORA_PATHS.keys())[0]
        first_path = LORA_PATHS[first_K]
        model = PeftModel.from_pretrained(base_model, first_path,
                                          adapter_name=get_adapter_name_for_K(first_K))
        # 3) 再把剩下的 LoRA 也挂上去
        for K, lora_path in LORA_PATHS.items():
            if K == first_K:
                continue
            model.load_adapter(lora_path, adapter_name=get_adapter_name_for_K(K))

        # 默认先用first_K对应的adapter
        model.set_adapter(get_adapter_name_for_K(first_K))
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            local_files_only=True,
            dtype=args.dtype,
        )

    # 推理模式加速
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def evaluate_vanilla_fast(args, model, tokenizer, eval_dataset):
    """
    只判断输出是否符合基础格式，开头和结尾是否是中括号，列表中元素的数量是否等于2*k*Nt，不对其他约束做额外的判断
    因此，该函数的可行性判断只有0和1，完全可行和完全不可行
    由于是多lora场景的评估，因此默认batch_size=1
    """
    subset = selected_eval_dataset(args.num_samples, eval_dataset)
    mse_list = []
    valid_list = []

    model.eval()

    # 按 batch 处理
    for i in tqdm(range(0, len(subset)), desc="Evaluating vanilla"):

        sample = subset[i]
        prompt = sample["prompt"]
        input_obj = sample["input"]
        K = int(sample["num_users"])
        CRB_gt = float(sample["objective"])

        # 根据 K 切换 LoRA
        set_adapter_for_K(model, K)

        # tokenizer批处理，因为单样本，此时不需要padding了
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)

        # 批量测试
        with torch.no_grad():
            gen_out = model.generate(
                **inputs, temperature=args.temperature,
                max_new_tokens=args.max_completion_length, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

        # 只取新生成的部分，去掉prompt
        gen_tokens = gen_out[:, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        # 解析输出
        status, llm_output = parse_w_from_pred(gen_text, K, config)

        if status == FORMAT_ERROR or llm_output is None:
            # 解析失败直接认为当前样本不可行
            valid_list.append(0)
            continue

        W_stack = vectors_to_W_stack(llm_output, config, K)
        valid_list.append(1)

        theta = float(input_obj["theta"])
        CRB_pred = compute_crb_for_sample(config, theta, W_stack)

        diff = float(CRB_pred) - float(CRB_gt)
        mse = diff * diff
        mse_list.append(round(mse, 4))

    return mse_list, valid_list


def evaluate_vanilla_all_conditions(args, model, tokenizer, eval_dataset):
    """
    判断所有的约束条件，类似于rl里的可行性奖励函数reward_feas_func_isac，采用加权分数判断是否可行
    """
    subset = selected_eval_dataset(args.num_samples, eval_dataset)
    mse_list = []
    valid_list = []

    model.eval()

    # 按 batch 处理
    for i in tqdm(range(0, len(subset)), desc="Evaluating vanilla (all conditions)"):

        sample = subset[i]
        prompt = sample["prompt"]
        input_obj = sample["input"]
        K = int(sample["num_users"])
        CRB_gt = float(sample["objective"])

        valid_score = 0

        # 根据 K 切换 LoRA
        set_adapter_for_K(model, K)

        # tokenizer批处理，因为单样本，此时不需要padding了
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)

        # 批量测试
        with torch.no_grad():
            gen_out = model.generate(
                **inputs, temperature=args.temperature,
                max_new_tokens=args.max_completion_length, do_sample=False,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

        # 只取新生成的部分，去掉prompt
        gen_tokens = gen_out[:, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

        # 解析输出
        status, llm_output = parse_w_from_pred(gen_text, K, config)

        if status == FORMAT_ERROR or llm_output is None:
            # 解析失败直接认为当前样本不可行
            valid_score = 0.0
            valid_list.append(valid_score)
            continue

        valid_score += FEAS_WEIGHTS["parse"]

        # 格式正确，但是维度错误，parse_w_from_pred自动补全了维度
        if status == FEASIBILITY_ERROR:
            valid_list.append(valid_score)
            continue

        valid_score += FEAS_WEIGHTS["shape"]

        W_stack = vectors_to_W_stack(llm_output, config, K)

        power_ratio = feasibility_power(W_stack, input_obj["PT"], K)
        valid_score += FEAS_WEIGHTS["power"] * power_ratio

        H = compute_channel_H(input_obj)
        sinr_ratio = feasibility_SINR(K, config, H, input_obj["Gamma"], W_stack)
        valid_score += FEAS_WEIGHTS["sinr"] * sinr_ratio

        valid_list.append(valid_score)

        theta = float(input_obj["theta"])
        CRB_pred = compute_crb_for_sample(config, theta, W_stack)

        diff = float(CRB_pred) - float(CRB_gt)
        mse = diff * diff
        mse_list.append(round(mse, 4))

    return mse_list, valid_list


def evaluate_best_of_n_fast(args, model, tokenizer, eval_dataset):
    """
    evaluate_vanilla_fast的best-of-n 评估版本
    对每个样本采样n个输出，选CRB最小的那个作为该样本的表现。
    """
    subset = selected_eval_dataset(args.num_samples, eval_dataset)

    mse_list = []
    valid_list = []
    n_candidates = args.best_of_n

    model.eval()

    for i in tqdm(range(0, len(subset)), desc="Evaluating best-of-n"):
        sample = subset[i]
        prompt = sample["prompt"]
        input_obj = sample["input"]
        K = int(sample["num_users"])
        CRB_gt = float(sample["objective"])

        # 根据K切换LoRA
        set_adapter_for_K(model, K)

        # 单样本tokenization
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(model.device)

        # 批量采样n个completion
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=n_candidates,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_tokens = gen_out[:, inputs["input_ids"].shape[1]:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        best_CRB = None
        best_mse = None
        # 候选样本中是否一个一个可行的
        any_valid = False

        # 对生成的n个样本逐个判断
        for gen_text in gen_texts:
            status, llm_output = parse_w_from_pred(gen_text, K, config)
            if status == FORMAT_ERROR or llm_output is None:
                # 这个候选无效，但不立即放弃整个样本，会继续判断后续的可选样本
                continue

            # 有一个可行的就标记为true
            any_valid = True
            W_stack = vectors_to_W_stack(llm_output, config, K)

            theta = float(input_obj["theta"])
            CRB_pred = compute_crb_for_sample(config, theta, W_stack)
            if (best_CRB is None) or (CRB_pred < best_CRB):
                best_CRB = CRB_pred
                diff = float(CRB_pred) - CRB_gt
                best_mse = diff * diff

        # 有可行的样本标记为1，否则就标记为0
        if not any_valid:
            # 这一条样本的n个候选全部失败
            valid_list.append(0)
        else:
            valid_list.append(1)
            mse_list.append(round(best_mse, 4))

    return mse_list, valid_list


def evaluate_best_of_n_all_conditions(args, model, tokenizer, eval_dataset):
    subset = selected_eval_dataset(args.num_samples, eval_dataset)

    mse_list = []
    valid_list = []
    n_candidates = args.best_of_n

    model.eval()

    for i in tqdm(range(0, len(subset)), desc="Evaluating best-of-n (all conditions)"):
        sample = subset[i]
        prompt = sample["prompt"]
        input_obj = sample["input"]
        K = int(sample["num_users"])
        CRB_gt = float(sample["objective"])

        valid_score = 0
        # 根据 K 切换 LoRA
        set_adapter_for_K(model, K)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=args.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=n_candidates,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # gen_out: [current_bs * n_candidates, prompt_len + new_len]
        gen_tokens = gen_out[:, inputs["input_ids"].shape[1]:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        best_CRB = None
        best_mse = None
        valid_of_best_CRB = None  # 与best_CRB对应的valid_score
        best_valid_any = 0.0  # 所有候选里最高的valid_score

        # 对生成的n个样本逐个判断
        for gen_text in gen_texts:
            status, llm_output = parse_w_from_pred(gen_text, K, config)
            if status == FORMAT_ERROR or llm_output is None:
                # 完全解析失败，valid_score = 0
                cand_valid = 0.0
                best_valid_any = max(best_valid_any, cand_valid)
                continue

            cand_valid = 0.0
            cand_valid += FEAS_WEIGHTS["parse"]


            if status == FEASIBILITY_ERROR:
                # 维度有问题，parse_w_from_pred做了补全/截断
                # 虽然不算 CRB，但保留这个candidate的 valid_score，用于计算best_valid_any
                best_valid_any = max(best_valid_any, cand_valid)
                continue

            # 维度
            cand_valid += FEAS_WEIGHTS["shape"]

            W_stack = vectors_to_W_stack(llm_output, config, K)

            # 功率约束
            power_ratio = feasibility_power(W_stack, input_obj["PT"], K)
            cand_valid += FEAS_WEIGHTS["power"] * power_ratio

            # SINR 约束
            H = compute_channel_H(input_obj)
            sinr_ratio = feasibility_SINR(K, config, H, input_obj["Gamma"], W_stack)
            cand_valid += FEAS_WEIGHTS["sinr"] * sinr_ratio

            # 更新best_valid_any
            best_valid_any = max(best_valid_any, cand_valid)

            # 对这个 candidate 计算 CRB
            theta = float(input_obj["theta"])
            CRB_pred = compute_crb_for_sample(config, theta, W_stack)
            diff = float(CRB_pred) - CRB_gt
            cand_mse = diff * diff

            # best输出的选择仍然是CRB最小，而不是可行性最高
            if (best_CRB is None) or (CRB_pred < best_CRB):
                best_CRB = CRB_pred
                best_mse = cand_mse
                valid_of_best_CRB = cand_valid

        # 处理这个样本的最终记录
        if best_CRB is not None:
            # 有至少一个candidate能算CRB
            valid_list.append(valid_of_best_CRB)
            mse_list.append(round(best_mse, 4))
        else:
            # 没有candidate能算CRB，则只记录最佳valid_score
            valid_list.append(best_valid_any)

    return mse_list, valid_list


def evaluate_model(args):
    model, tokenizer = load_model_and_tokenizer(args)

    eval_dataset = get_dataset(args.dataset_eval_path)

    if args.eval_method == 'vanilla_fast':
        return evaluate_vanilla_fast(args, model, tokenizer, eval_dataset)
    elif args.eval_method == 'vanilla_conditions':
        return evaluate_vanilla_all_conditions(args, model, tokenizer, eval_dataset)
    elif args.eval_method == 'best_of_n_fast':
        return evaluate_best_of_n_fast(args, model, tokenizer, eval_dataset)
    else:
        return evaluate_best_of_n_all_conditions(args, model, tokenizer, eval_dataset)


def main(args):

    set_global_seed(args.seed)

    print(f"Running multi-lord adapter")
    print(f"Number of samples: {args.num_samples}")

    mse_list, valid_list = evaluate_model(args)

    # 总体可行比例
    if mse_list:
        overall_mse = float(np.mean(mse_list))
    else:
        overall_mse = float("nan")

    overall_valid = float(np.mean(valid_list))

    print(f"\nOverall valid percent for {args.num_samples}")
    print(f"valid_ratio = {overall_valid: .3f}, MSE = {overall_mse: .3f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
