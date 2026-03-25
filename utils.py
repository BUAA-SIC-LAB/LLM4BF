import json
from datasets import load_dataset
import ast
from isac_utils import *
import re
import random

# 模板，用于提示模型和判断模型输出的位置
SYSTEM_PROMPT = (
    "You are a large language model specialized in solving downlink ISAC beamforming "
    "Below is an instruction describing this optimization problem. "
)

RESPONSE_TEMPLATE = "\n# Response:\n"

SUCCESS = 0
FORMAT_ERROR = 1
FEASIBILITY_ERROR = 2

# 数据的可行性权重
FEAS_WEIGHTS = {
    "parse": 0.5,  # 能被解析出来
    "shape": 0.3,  # 用户数与向量长度匹配情况
    "sinr": 0.1,  # 满足 SINR 约束
    "power": 0.1,  # 满足功率约束
}


def round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [round_floats(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    return obj


def get_dataset(data_dir):
    """
    可用于加载验证集或RL训练，因为该数据集不包含标签：
    Prompt截止到RESPONSE_TEMPLATE，不包含 Output。
    同时保留原始的output字段以便计算指标。
    """
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        user_nums = examples["num_users"]

        # 仅仅用于生成Prompt，不包含答案
        prompts = []
        for instruction, input_text, user_num in zip(instructions, inputs, user_nums):
            # 对输入浮点数做圆整处理
            input_rounded = round_floats(input_text, ndigits=3)
            input_str = json.dumps(input_rounded, ensure_ascii=False, separators=(",", ":"))

            # 构造 Prompt：只到 "# Response:\n" 为止
            text = (
                    SYSTEM_PROMPT
                    + "\n# Users:\n"
                    + str(user_num)
                    + "\n# Instruction:\n"
                    + instruction
                    + "\n# Input:\n"
                    + input_str
                    + RESPONSE_TEMPLATE
            )
            prompts.append(text)

        return {"prompt": prompts}  # 返回 prompt 字段供生成使用

    raw_datasets = load_dataset("json", data_files=data_dir, split="train")

    # map处理，不需要像训练集那样转成text并mask，因为我们是用来做generation的
    map_dataset = raw_datasets.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        remove_columns=[]
    )

    return map_dataset


def parse_w_from_pred(gen_text, K, config):
    """
    将模型的输出转化为向量w_k，返回三种状态码代表解析的结果
    """
    # 用正则表达式匹配字符，最外层的 \[ ... \] 是匹配中括号本身
    # ()在正则里表示捕获组，把这一段记下来，匹配成功之后可以用group(1)把这块内容取出来
    # [...]表示字符集合，^表示取反，^\]表示匹配除了]以外的所有字符，+表示重复多次
    match = re.search(r"\[([^\]]+)\]$", gen_text)
    if not match:
        return FORMAT_ERROR, None

    # 取出中括号内部的字符串，不包含中括号本身
    inner = match.group(1)

    # 按逗号切分，去掉收尾空格后转float
    raw_tokens = inner.split(",")
    values = []
    for token in raw_tokens:
        token = token.strip()
        if token == "":
            # 空串直接跳过
            continue
        try:
            values.append(float(token))
        except ValueError:
            # 只要有一个不是数字，就认为格式错误
            return FORMAT_ERROR, None

    # 长度检查，进行补零或截断
    Nt = config.Nt
    expect_length = K * 2 * Nt
    parsed_len = len(values)
    length_fixed = (parsed_len != expect_length)

    if parsed_len >= expect_length:
        # 太长，截到预期长度
        llm_output = values[:expect_length]
    else:
        # 太短，在末尾补0
        llm_output = values + [0.0] * (expect_length - parsed_len)

    status = SUCCESS if not length_fixed else FEASIBILITY_ERROR
    return status, llm_output


def feasibility_power(W_stack, PT, K, power_tolerance=1e-1):
    """
    检查约束问题的功率可行性，总功率 = sum_k real(tr(W_k))
    """
    total_power = 0.0
    for k in range(K):
        total_power += float(np.real(np.trace(W_stack[k])))

    if total_power <= PT + power_tolerance:
        power_ratio = 1.0
    else:
        # 超功率则按比例打折
        power_ratio = max(0.0, min(1.0, PT / (total_power + 1e-12)))
    return power_ratio


def feasibility_SINR(K, config, H, Gamma_dB, W_stack, sinr_tolerance=1e-1):
    """
    检查约束问题的SINR可行性
    """
    Gamma = config.Gamma(Gamma_dB)
    Nt = config.Nt

    Q = np.zeros((K, Nt, Nt), dtype=np.complex128)
    for k in range(K):
        hk = H[k].reshape(Nt, 1)
        Q[k] = hk @ hk.conj().T

    sinr_satisfied = 0
    for k in range(K):
        Qk = Q[k]

        # 信号项：tr(Q_k W_k)
        signal_k = float(np.real(np.trace(Qk @ W_stack[k])))

        # 干扰项：sum_{j≠k} tr(Q_k W_j)
        interf_k = 0.0
        for j in range(K):
            if j == k:
                continue
            interf_k += float(np.real(np.trace(Qk @ W_stack[j])))

        lhs = signal_k - Gamma * interf_k
        rhs = Gamma * config.sigma2C

        if lhs + sinr_tolerance >= rhs:
            sinr_satisfied += 1

    if K > 0:
        sinr_ratio = sinr_satisfied / K
    else:
        sinr_ratio = 0.0
    return sinr_ratio


def selected_eval_dataset(num_samples, eval_dataset):
    """
    随机选取验证集中的n个样本用于模型的测试
    """
    n = min(num_samples, len(eval_dataset))
    indices = random.sample(range(len(eval_dataset)), n)
    return eval_dataset.select(indices)
