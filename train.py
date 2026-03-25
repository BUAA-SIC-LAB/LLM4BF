import argparse
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
import warnings
from datetime import datetime
import torch
from utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Trainer for solving ISAC optimization problem')

    # Model and data parameters
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'],
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                        help='Use 4-bit quantization to reduce memory usage')
    parser.add_argument('--local_file', action='store_true', default=True,
                        help='Load LLM pretrained checkpoint from local path')
    parser.add_argument('--model_name', type=str,
                        default='./model_cache/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit', help='Model name')
    parser.add_argument('--data_dir', type=str,
                        default='dataset/ISAC_Dataset_Wk_1_SFT.json', help='Dataset path')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Scaling factor for LoRA updates')
    parser.add_argument('--bias', type=str, default='lora_only', choices=['none', 'all', 'lora_only'],
                        help='Bias type, if use eval_adapter, set this options **none**')
    parser.add_argument('--use_rslora', action='store_true', default=False, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')

    # Training configurations
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=20, help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=50, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=500, help='Steps interval to save model checkpoints')
    parser.add_argument('--train_lm_head', action='store_true', default=False,
                        help='Whether to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true', default=False,
                        help='Whether to train the embed_tokens or not')
    parser.add_argument('--output_dir', type=str, default='./history', help='Output directory name')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='./history/20251218_001834/output_alpha16_r16_seq2048_batch16_epoch4/checkpoint-2000')

    args = parser.parse_args()

    return args


# Initialize the system model parameters for the communication scenario.
config = ISACConfig()


class SafeDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    应用在训练阶段，给LLM的输出加mask，让模型只在回答的部分计算loss：
    输入格式：prompt的固定模板（Instruction + Input + Response + EOS）
    通过匹配模板中Response在token序列中的位置，把它前面的token全部置为-100，从而将注意力权重压到最低
    """

    def __init__(self, response_template, tokenizer, mlm=False, fallback_strategy="last_portion"):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        # 设置回答的模板，函数会寻找这个模板，并把之后的内容当做模型的预测输出
        # 将模板前面内容设置为-100，计算后面模型输出的loss
        self.response_template = response_template
        self.tokenizer = tokenizer

        # 把回答的模板转换成token序列
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.fallback_strategy = fallback_strategy

    def torch_call(self, examples):
        # 父类方法接收一批examples，将输入复制一份得到label，然后padding到相同的长度
        # 在tokenizer部分，数据会被处理成字典的格式，其中input_ids代表数据集输入
        # input_ids里面是很多的列表，代表多条转换成token的数据，例如
        # batch["input_ids"] = [
        #     [101, 200, 300, 999, 500],  # 第0条数据 (i=0)
        #     [101, 200, 305, 999, 600]  #  第1条数据 (i=1)
        # ]
        # super().torch_call还把输入复制了一遍，生成了batch["labels"]，和input_ids一模一样，用于标注掩码
        batch = super().torch_call(examples)

        # 遍历batch里的每个样本，len(batch["input_ids"])就是batch size
        for i in range(len(batch["input_ids"])):
            # 取出一条数据
            input_ids = batch["input_ids"][i].tolist()

            # decode 回完整字符串
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            # 在字符串里找模板
            char_pos = text.find(self.response_template)

            if char_pos != -1:
                # 找到模板，取出到模板末尾的前缀
                prefix_text = text[: char_pos + len(self.response_template)]

                # 对前缀重新编码，得到前缀token长度
                prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                response_start_idx = len(prefix_ids)

                # mask模板及其之前的 labels
                batch["labels"][i, :response_start_idx] = -100
                continue
            # ====== 找不到模板：兜底策略 ======
            warnings.warn(f"Response template not found in example {i}")

            if self.fallback_strategy == "last_portion":
                seq_length = len(input_ids)
                start_pos = int(0.9 * seq_length)
                batch["labels"][i, :start_pos] = -100

            elif self.fallback_strategy == "full_example":
                pass  # 不做mask，整句训练

            elif self.fallback_strategy == "skip":
                batch["labels"][i, :] = -100

        return batch


def get_isac_sft_datasets(tokenizer, data_files):
    """
    构造 SFT 用的数据集：
    将json文件的数据转换为固定的格式，以'text'的形式输出，包含：
    SYSTEM_PROMPT, Instruction, Input(json), RESPONSE_TEMPLATE, Output, EOS
    """
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        # 从json中取出相应的字段
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        user_nums = examples["num_users"]

        # 最后给模型输入的prompt放在text里
        texts = []
        for instruction, input_text, output_text, user_num in zip(
                instructions, inputs, outputs, user_nums
        ):
            # 在这里把所有浮点数统一保留3位
            input_rounded = round_floats(input_text, ndigits=3)
            output_rounded = round_floats(output_text, ndigits=3)

            # 转成紧凑JSON，减少token长度，方便LLM解析
            input_str = json.dumps(input_rounded, ensure_ascii=False, separators=(",", ":"))
            output_str = json.dumps(output_rounded, ensure_ascii=False, separators=(",", ":"))

            # prompt = SYSTEM_PROMPT + user_num + Instruction + Input + RESPONSE_TEMPLATE + Output + EOS
            text = (
                    SYSTEM_PROMPT
                    + "\n# Users:\n"
                    + str(user_num)
                    + "\n# Instruction:\n"
                    + instruction
                    + "\n# Input:\n"
                    + input_str
                    + RESPONSE_TEMPLATE
                    + output_str
                    + EOS_TOKEN
            )
            texts.append(text)

        return {"text": texts}

    raw_datasets = load_dataset("json", data_files=data_files, split="train")

    train_dataset = raw_datasets.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        remove_columns=[],    # 不删除json数据中其他的字段
    )

    # 方便外面直接拿到 collator
    data_collator = SafeDataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
        fallback_strategy="last_portion",
    )

    return train_dataset, data_collator


def get_isac_sft_datasets_numeric_only(tokenizer, data_files):
    """
    纯数值版本的SFT数据集构造：
    不再包含 SYSTEM_PROMPT / Instruction / JSON 键名 等自然语言。
    """
    EOS_TOKEN = tokenizer.eos_token

    def flatten_2d(mat):
        return [x for row in mat for x in row]

    def fmt(v):
        # 统一保留 3 位小数
        return f"{float(v): .3f}"

    def formatting_prompts_func(examples):
        inputs = examples["input"]          # List[dict]
        outputs = examples["output"]        # List[List[float]]
        user_nums = examples["num_users"]   # List[int or float]

        texts = []
        for inp, out, K in zip(inputs, outputs, user_nums):
            # 1) 从 input 里取出各个字段
            theta = inp["theta"]
            PT = inp["PT"]
            Gamma = inp["Gamma"]
            H_real = inp["H_real"]   # 形状 [K, Nt]
            H_imag = inp["H_imag"]   # 形状 [K, Nt]

            # 2) 展平成一维向量： [num_users, theta, PT, Gamma, H_real_flat..., H_imag_flat...]
            H_real_flat = flatten_2d(H_real)
            H_imag_flat = flatten_2d(H_imag)

            x_vec = [K, theta, PT, Gamma, *H_real_flat, *H_imag_flat]
            y_vec = out

            # 3) 转成字符串（纯数字 + 空格）
            input_str = " ".join(fmt(v) for v in x_vec)
            output_str = " ".join(fmt(v) for v in y_vec)

            # 4) 拼成最终 text：
            #    <纯数值输入> + RESPONSE_TEMPLATE + <纯数值输出> + EOS
            text = input_str + RESPONSE_TEMPLATE + output_str + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    raw_datasets = load_dataset("json", data_files=data_files, split="train")

    train_dataset = raw_datasets.map(
        formatting_prompts_func,
        batched=True,
        load_from_cache_file=False,
        remove_columns=[]
    )

    data_collator = SafeDataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
        fallback_strategy="last_portion",
    )

    return train_dataset, data_collator


def train_model(args):
    tensor_dir = "tensor_log"

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        local_files_only=args.local_file,
        cache_dir="./cache",
        dtype=dtype
    )

    train_dataset, data_collator = get_isac_sft_datasets(tokenizer, args.data_dir)

    if args.resume_from_checkpoint is not None:
        checkpoint_path = args.resume_from_checkpoint.rstrip("/")
        # 实验主目录
        dir_out = os.path.dirname(checkpoint_path)
        # run根目录
        run_root = os.path.dirname(dir_out)
        # tensorboard
        dir_log = os.path.join(run_root, tensor_dir)
    else:
        # 正常开一个新 run
        run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_format_dir = (f"output_alpha{args.lora_alpha}_r{args.lora_r}_"
                          f"data_length{len(train_dataset)}_"
                          f"batch{args.batch_size}_"
                          f"epoch{args.num_train_epochs}")

        dir_out = os.path.join(args.output_dir, run_time, log_format_dir)
        dir_log = os.path.join(args.output_dir, run_time, tensor_dir)

    # 设置LoRA的位置
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # 输出层
    if args.train_lm_head:
        target_modules.append('lm_head')
    # embedding层
    if args.train_embed_tokens:
        target_modules.append('embed_tokens')

    # 这里在设置LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,  # LoRA的秩
        target_modules=target_modules,  # 在哪些地方添加LoRA
        lora_alpha=args.lora_alpha,  # LoRA的缩放系数
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,  # 为了省显存，把前向传播的一部分中间结果丢掉，反向传播时再现算一遍
        random_state=args.seed,
        use_rslora=args.use_rslora,  # 一种改进算法，把lora的结果除以根号r，保证训练的稳定
        loftq_config=args.loftq_config
    )

    # SFT的训练过程如下：
    # 1. 调用tokenizer把待输入的数据转换成batch组成的文本列表，里面包含了"input_ids"
    # 2. 调用collator(SafeDataCollatorForCompletionOnlyLM)，padding到相同长度，复制一份input_ids到labels，并根据模板把描述mask成-100
    # 3. 把batch放入model里面训练，计算loss，更新模型
    # 4. 可选的操作：梯度累积，调整学习率，checkpoint保存等。
    sft_args = SFTConfig(
        output_dir=dir_out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_step,
        save_total_limit=args.save_total_limit,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        dataset_num_proc=16,
        packing=False,
        report_to=["tensorboard"],
        logging_dir=dir_log,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        args=sft_args,
    )

    if args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    return trainer


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
