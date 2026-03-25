# LLM4BF: LLM for ISAC Beamforming

Official repository for paper **From General LLM to Specialized Beamforming for Low-Altitude ISAC Networks: A LoRA-Based Multi-Expert Framework**.

This is a general framework based on LLMs. We adopt the Unsloth framework, which significantly reduces GPU memory usage and improves training speed. Different backbone models can be selected according to specific requirements. Furthermore, the communication model can be designed for particular task scenarios, with customized data formats and corresponding data generation. 

**Training based on this framework enables LLM empowered wireless communication.**

![algorithm](figures/algorithm.jpg)



## Absrtact

Integrated sensing and communication (ISAC) achieves both radar sensing and data transmission, yet the design of ISAC beamforming often leads to inherently non-convex optimization problems. Recently, large language models (LLMs), as representatives of general artificial intelligence (AI), have exhibited remarkable capabilities in mathematical reasoning and problem-solving. Empowering ISAC with LLM has thus attracted considerable attention as a promising direction. However, most existing studies simply treat a pretrained LLM as a black-box solver without further adaptation, failing to adapt a general-purpose model into a task-oriented expert. In this paper, we pioneer the use of LLMs to construct an end-to-end ISAC optimizer. We design a multi-expert framework that enables adaptation to diverse ISAC scenarios. Leveraging the strength of LLM in natural language (NL) understanding, we reformulate communication scenarios into NL descriptions, thereby eliminating the need for preprocessing modules and explicit mathematical derivations. Furthermore, inspired by the mixture-of-experts paradigm, we employ low-rank adaptation to specialize the LLM for ISAC optimization, enabling it to generalize across diverse scenarios. We develop a two-stage training framework. In the supervised fine-tuning stage, the model learns to generate solutions with a structured output format. Then, reinforcement learning further refines the outputs to ensure constraint feasibility and numerical optimality. Extensive experiments demonstrate that our approach achieves superior performance, significantly outperforming existing AI-based methods.



## Code

This study mainly consists of two stages, namely Supervised Fine-Tuning (SFT), and Reinforcement Learning (RL), which correspond to `tarin.py` and `rl.py`, respectively. Following the order presented in the paper, `tarin.py` is executed first to save the trained LoRA, which is then loaded in `rl.py` for further training. `eval.py` is used to evaluate model performance, and this file only tests a single LoRA module. `eval_adapter.py` adopts a mixture of experts framework, which requires training multiple LoRA modules and loading them into the same backbone model. Since `eval.py` only considers one LoRA, the test data can be evaluated in batches, resulting in much faster evaluation. In contrast, `eval_adapter.py` only supports sequential evaluation.

The specific experimental configurations are defined in the code and can be adjusted as needed. We include necessary comments in the code to facilitate readability.



## Training

The training process consists of four main stages:

### 1. Environment Preparation

We use the [Unsloth](https://github.com/unslothai/unsloth) framework to implement efficient fine tuning of the LLM. Therefore, this framework needs to be installed. **Note that automatic installation via pip uninstalls the existing PyTorch environment and installs the latest version of PyTorch.**

The training is based on the `Qwen2.5-3B-Instruct-unsloth-bnb-4bit` backbone model, so this model needs to be downloaded. The official download method can be used, or third party tools can also be adopted.

~~~
hf download unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit --local-dir ./Qwen2.5-3B-Instruct-unsloth-bnb-4bit
modelscope download --model unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit --cache_dir ./model_cache
~~~



### 2. Supervised Fine-Tuning

Default parameters are configured in `tarin.py`, and it can be executed directly. Note that `--model_name` is modified to the path of the backbone model, and `--data_dir` is modified to the dataset path. The `--output_dir` argument specifies the location for logging, and logs are recorded using TensorBoard.



### 3. Reinforcement Learning 

Similarly, default parameters are configured in `rl.py`. If GPU memory is insufficient, `--max_prompt_length`, `--max_completion_length`, and `--num_generations` are adjusted accordingly. The `--model_path` argument specifies the LoRA module trained in the SFT stage and needs to be set to the corresponding path. The `--base_model` argument also needs to be set to the backbone model path.



### 4. Testing

If GPU memory is limited, `eval.py` is recommended for evaluation, which tests only a single LoRA module and achieves faster execution. Two evaluation modes are provided. The `vanilla` mode allows the LLM to generate only one solution, whereas `best_of_n` generates multiple solutions and selects the optimal one as the output.

`eval_adapter.py` loads multiple LoRA modules corresponding to different scenarios. The code automatically routes the input data to the appropriate LoRA to ensure accurate output.



## Data

![system_model](figures/system_model.jpg)

Our input data is derived from the analytical solution of the system model presented in the paper. In `isac_solve.m`, we implement the numerical solution in MATLAB. Therefore, `isac_solve.m` needs to be executed to generate the training data. The generated data is stored in `.mat` format. Subsequently, the `.mat` files need to be converted into `.json` files. The format of the `.json` files must strictly follow the specification described in the paper, otherwise the code logic in model training and evaluation requires corresponding modification. 

![dataset](figures/dataset.jpg)

The completely training data are released as open source after the paper is accepted.



## Weight

The completely trained model weights and execution logs are released as open source after the paper is accepted.



## Reference Work

Unsloth: https://github.com/unslothai/unsloth

LLMCoSolver: https://arxiv.org/abs/2509.16865

Cramér-Rao Bound Optimization: https://ieeexplore.ieee.org/abstract/document/9652071
