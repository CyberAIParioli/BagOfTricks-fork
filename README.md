<p align="center"><img src="assets/framework-v6.png" alt="framework_"/></p>
<div align="center">

# <img src="assets/bag.png" style="height: 1em; vertical-align: middle;" />Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs

</div>

- [Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs](#bag-of-tricks-benchmarking-of-jailbreak-attacks-on-llms)
  - [0. Overview](#0-overview)
    - [Update](#update)
  - [1. Quick Start](#1-quick-start)
    - [1.1 Installation](#11-installation)
    - [1.2 Preparation](#12-preparation)
      - [1.2.1 OPENAI API Key](#121-openai-api-key)
      - [1.2.2 Model Preparation](#122-model-preparation)
    - [1.3 One-Click Run](#13-one-click-run)
      - [1.3.1 Run different tricks](#131-run-different-tricks)
      - [1.3.2 Run main experiments](#132-run-main-experiments)
  - [2. Definition of Argument](#2-definition-of-argument)
  - [3. Supported Methods](#3-supported-methods)
    - [3.1 Attack Methods](#31-attack-methods)
  - [4. Supported LLMs](#4-supported-llms)
  - [5. Acknowledgement](#5-acknowledgement)

## 0. Overview

Although Large Language Models (LLMs) have demonstrated significant capabilities in executing complex tasks in a zero-shot manner, they are susceptible to jailbreak attacks and can be manipulated to produce harmful outputs. This repository provides a comprehensive benchmark for evaluating jailbreak attacks on LLMs. We evaluate various attack settings on LLM performance and provide a standardized evaluation framework. Specifically, we evaluate key factors of implementing jailbreak attacks on LLMs from both target-level and attack-level perspectives. 

### Update

1. **2024.08 Update1:** We support more LLMs including: Llama2, Llama3, Mistral, Qwen, Tulu, and Vicuna families. See all supported models and details [HERE](#4-supported-llms).

2. **2024.08 Update2:** We add two new attack methods: [DrAttack](https://arxiv.org/abs/2402.16914) and [MultiJail](https://arxiv.org/abs/2310.06474). 
   **DrAttack** is a new prompt-level jailbreak methods and need to preprocessing the data. We have provide the preprocessing results for datasets used in this repo. 
   **MultiJail** (ICLR 2024) is a new type of jailbreak method that manually-create multilingual prompts to attack the LLMs. There are 10 languages supported in MultiJail, including: English, High-Resource Language (HRL) : Chines (zh), Italic (it), Vietnamese (vi); Medium-Resource Language (MRL): Arabic (ar), Korean (ko), Thai (th); Low-Resource Language (LRL): Bengali (bn), Swahili (sw), Javanese (jv).

3. **2024.10 Update:** We are working on new version of **JailTrickBench** which will be released in the future. The new version will include more attack methods and will support more LLMs. Stay tuned!

🌟 **If you find this resource helpful, please consider starring this repository and citing our NeurIPS'24 paper:**

```
@inproceedings{NEURIPS2024_xu2024bag,
 author={Xu, Zhao and Liu, Fan and Liu, Hao},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs},
 year = {2024}
}

@article{xu2024bag,
  title={Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs},
  author={Xu, Zhao and Liu, Fan and Liu, Hao},
  journal={arXiv preprint arXiv:2406.09324},
  year={2024}
}
```


## 1. Quick Start

### 1.1 Installation

```bash
git clone <Repo Link> or download the zip file from anonymous github
cd ./Bag_of_Tricks_for_LLM_Jailbreaking-4E10/
pip install -r requirements.txt
```

### 1.2 Preparation

#### 1.2.1 OPENAI API Key

Several attack baselines use GPT services in their methods. Please search and replace `YOUR_KEY_HERE` with your OpenAI API key in the following files: `./baseline/TAP/language_models.py`, `./baseline/PAIR/language_models.py`, `./baseline/GPTFuzz/gptfuzzer/llm/llm.py`, `./baseline/AutoDAN/utils/opt_utils.py`.

After setting the API key, you can run the attack experiments with one-click scripts below.

#### 1.2.2 Model Preparation

To run the attack experiments, you may need to prepare the following models:

- Attack Models:
  - AdvPrompter: You need to train the AdvPrompter model to get the LoRA Adapter and merge the model. You can save the attack model to `./models/attack/`.
  - AmpleGCG: If you cannot load the huggingface model directly using the `AutoModelForCausalLM.from_pretrained()` function, you may need to download the attack model `osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat` and `osunlp/AmpleGCG-llama2-sourced-vicuna-7b` from Hugging Face first.

### 1.3 One-Click Run

#### 1.3.1 Run different tricks

For each trick, we provide a example script. You can run the script to reproduce the results in the paper. The script is in the `./scripts/` folder. You can run the script by the following command:

```bash
# 1. Attack Budget
bash scripts/1_trick_atk_budget_gcg.sh
bash scripts/1_trick_atk_budget_pair.sh

# 2. Attack Ability
bash scripts/2_trick_atk_ability_pair.sh

# 3. Attack Suffix Length
bash scripts/3_trick_atk_suffix_length.sh

# 4. Attack Intension
bash scripts/4_trick_atk_intension_autodan.sh
bash scripts/4_trick_atk_intension_pair.sh

# 5. Target Model Size
bash scripts/5_trick_target_size_autodan.sh
bash scripts/5_trick_target_size_pair.sh

# 6. Target Safety Fine-tuning Alignment
bash scripts/6_trick_target_align_autodan.sh
bash scripts/6_trick_target_align_pair.sh

# 7. Target System Prompt
bash scripts/7_trick_target_system_autodan.sh
bash scripts/7_trick_target_system_pair.sh

# 8. Target Template Type
bash scripts/8_trick_target_template_autodan.sh
bash scripts/8_trick_target_template_pair.sh
```

**Note**: As some baselines require a long time to run, we provide a feature to run the experiment in parallel (We use 50 A800 GPUs to accelerate the experiments) You can set the `--data_split` and `--data_split_total_num` to run the experiment in parallel. For example, you can set `--data_split_total_num 2` and `--data_split_idx 0` in the script to run the first half of the data, and set `--data_split_total_num 2` and `--data_split_idx 1` in the script to run the second half of the data. After all data is finished, the program will automatically merge the results.

#### 1.3.2 Run main experiments

```bash
# Example: Use vicuna as target model
# You can run the attack experiments by the following command:

# 1. Run AutoDAN attack
bash scripts/main_vicuna/1_data1_None_defense.sh

# 2. Run PAIR attack
bash scripts/main_vicuna/2_data1_None_defense.sh

# 3. Run TAP attack
bash scripts/main_vicuna/3_data1_None_defense.sh

# 4. Run GPTFuzz attack
bash scripts/main_vicuna/4_data1_None_defense.sh

# 5. Run GCG attack
bash scripts/main_vicuna/5_data1_None_defense.sh

# 6. Run AdvPrompter attack
bash scripts/main_vicuna/6_data1_None_defense.sh

# 7. Run AmpleGCG attack
bash scripts/main_vicuna/7_data1_None_defense.sh
```

**Note1**: As some baselines require a long time to run, we provide a feature to run the experiment in parallel. You can set the `--data_split` and `--data_split_total_num` to run the experiment in parallel. For example, you can set `--data_split_total_num 2` and `--data_split_idx 0` in the script to run the first half of the data, and set `--data_split_total_num 2` and `--data_split_idx 1` in the script to run the second half of the data. After all data is finished, the program will automatically merge the results.

**Note2**: You can write your own script to customize the attack experiments.

## 2. Definition of Argument

For detailed arguments and options, please refer to the `initialie_args.py` or help message of `main.py`.

```bash
python main.py -h
```

## 3. Supported Methods

### 3.1 Attack Methods

- [AutoDAN][R-AutoDAN]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack AutoDAN \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --resume_exp \
    --exp_name main_vicuna_attack
  ```
- [PAIR][R-PAIR]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack PAIR \
    --attack_model lmsys/vicuna-13b-v1.5 \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --resume_exp \
    --exp_name main_vicuna_attack
  ```
- [TAP][R-TAP]
  ```bash
    python -u main.py \
      --target_model_path lmsys/vicuna-13b-v1.5 \
      --attack TAP \
      --attack_model lmsys/vicuna-13b-v1.5 \
      --instructions_path ./data/harmful_bench_50.csv \
      --save_result_path ./exp_results/main_vicuna/ \
      --resume_exp \
      --exp_name main_vicuna_attack
  ```

- [GPTFuzz][R-GPTFuzz]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack GPTFuzz \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --resume_exp
  ```

- [GCG][R-GCG]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack GCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --resume_exp \
    --exp_name main_vicuna_attack
  ```

- [AdvPrompter][R-AdvPrompter]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack AdvPrompter \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --resume_exp \
    --exp_name main_vicuna_attack \
    --adv_prompter_model_path ./models/attack/advprompter_vicuna_7b
  ```

- [AmpleGCG][R-AmpleGCG]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack AmpleGCG \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --resume_exp \
    --exp_name main_vicuna_attack \
    --attack_source vicuna
  ```
- [DrAttack][R-DrAttack]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack DrAttack \
    --instructions_path ./data/harmful_bench_50.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --prompt_info_path ./baseline/DrAttack/dratk_data/attack_prompt_data/harmful_bench_test_info.json \
    --resume_exp \
    --exp_name main_vicuna_attack
  ```

- [MultiJail][R-MultiJail]
  ```bash
  python -u main.py \
    --target_model_path lmsys/vicuna-13b-v1.5 \
    --attack MultiJail \
    --instructions_path ./baseline/MultiJail/multijail_data/1_MultiJail_en.csv \
    --save_result_path ./exp_results/main_vicuna/ \
    --resume_exp \
    --exp_name main_vicuna_attack
  ```

| Model                                                             | Model size                       |
| ----------------------------------------------------------------- | -------------------------------- |
| [Llama](https://github.com/facebookresearch/llama)                | 7B/13B/33B/65B                   |
| [Llama 2](https://huggingface.co/meta-llama)                      | 7B/13B/70B                       |
| [Llama 3/Llama 3.1](https://huggingface.co/meta-llama)            | 8B/70B                           |
| [Mistral/Mixtral](https://huggingface.co/mistralai)               | 7B/8x7B/8x22B                    |
| [Qwen/Qwen1.5/Qwen2](https://huggingface.co/Qwen)                 | 0.5B/1.5B/4B/7B/14B/32B/72B/110B |
| [Vicuna](https://huggingface.co/lmsys)                            | 7B/13B                           |

For model size larger than 13B/14B, we use 4 bit quantization to reduce the memory usage.

## 5. Acknowledgement
In the implementation of this project, we have referred to the code from the following repositories or papers:

- Attack methods: GCG, AutoDAN, PAIR, TAP, GPTFuzz, AdvPrompter, AmpleGCG, DrAttack, MultiJail
  - [GCG][R-GCG]
  - [AutoDAN][R-AutoDAN]
  - [PAIR][R-PAIR]
  - [TAP][R-TAP]
  - [GPTFuzz][R-GPTFuzz]
  - [AdvPrompter][R-AdvPrompter]
  - [AmpleGCG][R-AmpleGCG]
  - [DrAttack][R-DrAttack]
  - [MultiJail][R-MultiJail]

[R-GCG]: https://github.com/llm-attacks/llm-attacks
[R-AutoDAN]: https://github.com/SheltonLiu-N/AutoDAN
[R-PAIR]: https://github.com/patrickrchao/JailbreakingLLMs
[R-TAP]: https://github.com/RICommunity/TAP
[R-GPTFuzz]: https://github.com/sherdencooper/GPTFuzz
[R-AdvPrompter]: https://github.com/facebookresearch/advprompter
[R-AmpleGCG]: https://github.com/OSU-NLP-Group/AmpleGCG
[R-DrAttack]: https://github.com/xirui-li/DrAttack/tree/main
[R-MultiJail]: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs