import json
from datasets import load_dataset
import ast
from isac_utils import *
import re
import random

# Template, used to guide the model and identify the position of the model output.
SYSTEM_PROMPT = (
    "You are a large language model specialized in solving downlink ISAC beamforming "
    "Below is an instruction describing this optimization problem. "
)

RESPONSE_TEMPLATE = "\n# Response:\n"

SUCCESS = 0
FORMAT_ERROR = 1
FEASIBILITY_ERROR = 2

# The feasibility weight of the data.
FEAS_WEIGHTS = {
    "parse": 0.5,  # can be parsed
    "shape": 0.3,  # correlation between the number of users and vector length
    "sinr": 0.1,  # SINR constraint
    "power": 0.1,  # power constraint
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
    Can be used to load the validation set or for RL training, because this dataset does not contain labels:
    The prompt ends at RESPONSE_TEMPLATE and does not include Output.
    At the same time, the original output field is kept for metric calculation.
    """
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        user_nums = examples["num_users"]

        # Only used to generate the prompt, without including the answer.
        prompts = []
        for instruction, input_text, user_num in zip(instructions, inputs, user_nums):
            # Round the input floating-point numbers.
            input_rounded = round_floats(input_text, ndigits=3)
            input_str = json.dumps(input_rounded, ensure_ascii=False, separators=(",", ":"))

            # Build the prompt so that it only goes up to "# Response:\n".
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

        return {"prompt": prompts}

    raw_datasets = load_dataset("json", data_files=data_dir, split="train")

    # Use map processing. Unlike the training set,
    # it does not need to be converted into text and masked, because it is used for generation.
    map_dataset = raw_datasets.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        remove_columns=[]
    )

    return map_dataset


def parse_w_from_pred(gen_text, K, config):
    """
    Convert the model output into the vector w_k, and return three status codes to represent the parsing result.
    """
    match = re.search(r"\[([^\]]+)\]$", gen_text)
    if not match:
        return FORMAT_ERROR, None

    # Take out the string inside the square brackets, without the brackets themselves.
    inner = match.group(1)

    # Split it by commas, remove leading and trailing spaces, and then convert it to float.
    raw_tokens = inner.split(",")
    values = []
    for token in raw_tokens:
        token = token.strip()
        if token == "":
            continue
        try:
            values.append(float(token))
        except ValueError:
            return FORMAT_ERROR, None

    # Length check, pad with zeros or truncate
    Nt = config.Nt
    expect_length = K * 2 * Nt
    parsed_len = len(values)
    length_fixed = (parsed_len != expect_length)

    if parsed_len >= expect_length:
        # Too long, truncated to the expected length
        llm_output = values[:expect_length]
    else:
        # Too short, zeros added at the end
        llm_output = values + [0.0] * (expect_length - parsed_len)

    status = SUCCESS if not length_fixed else FEASIBILITY_ERROR
    return status, llm_output


def feasibility_power(W_stack, PT, K, power_tolerance=1e-1):
    """
    Check the power feasibility of the constrained problem, where the total power is sum_k real(tr(W_k)).
    """
    total_power = 0.0
    for k in range(K):
        total_power += float(np.real(np.trace(W_stack[k])))

    if total_power <= PT + power_tolerance:
        power_ratio = 1.0
    else:
        power_ratio = max(0.0, min(1.0, PT / (total_power + 1e-12)))
    return power_ratio


def feasibility_SINR(K, config, H, Gamma_dB, W_stack, sinr_tolerance=1e-1):
    """
    Check the SINR feasibility of the constrained problem.
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

        # Signal term: tr(Q_k W_k)
        signal_k = float(np.real(np.trace(Qk @ W_stack[k])))

        # Interference term: sum_{j≠k} tr(Q_k W_j)
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
    n = min(num_samples, len(eval_dataset))
    indices = random.sample(range(len(eval_dataset)), n)
    return eval_dataset.select(indices)
