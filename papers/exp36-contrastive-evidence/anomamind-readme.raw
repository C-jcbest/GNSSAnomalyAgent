<div align="center">
  <h1><img src="assets/image.png" alt="MemCast logo" style="height: 1em; width: auto; vertical-align: -0.15em; margin-right: 0.4em;">AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning</h1>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/AnomaMind-TS/stargazers">
    <img src="https://img.shields.io/github/stars/Xiaoyu-Tao/AnomaMind-TS?style=social" alt="Stars">
  </a>
  <a href="https://github.com/Xiaoyu-Tao/AnomaMind-TS/pulls">
    <img src="https://img.shields.io/badge/PRs-Welcome-green" alt="PRs Welcome">
  </a>
</div>

---

AnomaMind is a novel framework that reformulates **Time Series Anomaly Detection (TSAD)** as an **evidence-driven sequential decision-making process**. It combines a structured coarse-to-fine workflow with tool-augmented reasoning and a hybrid inference mechanism that couples general-purpose LLM reasoning with task-specific anomaly decision learning via reinforcement learning.

> 📝 "AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning"  
> **Preprint** | [📄 Paper]()

---

## 🔍 Overview

Existing TSAD methods often treat anomaly detection as a static discriminative task with fixed feature inputs, lacking adaptive evidence gathering and iterative refinement. AnomaMind introduces an agentic paradigm:

- **Coarse-to-Fine Workflow**: Progressively localizes anomalous intervals (coarse evidence acquisition), then performs adaptive evidence construction, reasoning-based detection, and iterative refinement.
- **Tool-Augmented Reasoning**: Relies on a **Detection Toolkit**—interval localization (visual-aware), feature extraction (statistical and structural), and knowledge memory—for context-aware diagnostic analysis.
- **Hybrid Inference**: General-purpose LLMs handle autonomous tool invocation and self-reflective refinement; core anomaly detection decisions are learned through **reinforcement learning** under workflow-level feedback.

<p align="center">
  <img src="assets/main.png" width="800">
</p>

---

## ✨ Key Features

- **Structured Coarse-to-Fine Workflow**: Coarse evidence acquisition → adaptive evidence construction → reasoning-based anomaly detection → iterative refinement.
- **Detection Toolkit**: Interval localization (VLM-based), statistical and structural feature extraction, and knowledge memory for domain and tool semantics.
- **Hybrid Decision Mechanism**: General-purpose models for tool orchestration and self-reflection; RL-optimized detector for precise anomaly boundaries.
- **Self-Reflective Refinement**: Evaluator validates intermediate results and triggers refinement when needed, reducing false alarms and improving consistency.
- **Strong Benchmark Performance**: Consistently improves over statistical, deep learning, foundation-model, and LLM-based baselines on YAHOO, TODS, IOPS, and WSD.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/ustc-time-series/AnomaMind.git
cd AnomaMind
```

### 2. Environment Setup

```bash
conda create -n anomalmind python=3.10 -y
conda activate anomalmind
pip install -r requirements.txt
pip install -e . --no-build-isolation
```


### 3. Prepare Data

AnomaMind is evaluated on four TSAD benchmarks with diverse anomaly types (point, contextual, collective, sequence):
<p align="center">
  <img src="assets/dataset.png" width="800">
</p>

Datasets can be downloaded from [here](https://www.thedatum.org/datasets/TSB-AD-U.zip). 

#### Put data in place
Create the dataset under `data/<name>/`:

```text
data/<name>/
├── <name>.yaml
└── raw/
    ├── in-series/
    └── cross-series/
```

- `<name>.yaml`: dataset config
- `raw/in-series/`: CSV files used for train and test
- `raw/cross-series/`: extra CSV files used for train only

#### Write the YAML

Use this format:

```yaml
segment_size: 100
processed_train_anomaly_ratio: 0.7
processed_test_anomaly_ratio: 0
```

#### Prepare the CSV files

Each CSV should contain the value column and `Label`.

- `in-series` files are split into train and test
- `cross-series` files are added to train only

#### Run the command

```bash
python -m AnomaMind.dataprep.build_dataset data/<name>
```

#### Output structure

After the command finishes, you will get:

```text
data/<name>/
├── raw/
│   ├── train/
│   └── test/
├── cluster/
└── processed/
    ├── train/
    ├── test/
    └── train_off/
```

### 4. SFT Training

#### Prepare env files

Create these files under `env/`:

- `actor.env`
- `detector.env`
- `locator.env`
- `evaluator.env`

Example:

```env
LLM_API_KEY=YOUR_LLM_KEY
LLM_BASE_URL=YOUR_LLM_URL
AUX_LLM_MODEL=YOUR_LLM_NAME
```

#### Run inference to collect SFT data

Use the processed train set as input and write results to `result/train`:

```bash
python -m AnomaMind.infer.infer \
  data/<name>/processed/train \
  --output-dir data/<name>/result/train \
  --workers 8 \
  --max-retries 3 \
  --no-reflection
```


#### Build SFT CSV

Merge multiple collected result folders into one cross-domain SFT CSV:

```bash
python -m AnomaMind.train.prepare_sft_data \
  --mode multi \
  --res_root data \
  --datasets KPI WSD IOPS YAHOO \
  --output_csv data/cross_domain/sft.csv \
  --balance upsample \
  --max_prompt_tokens 7000 \
  --tokenizer_path /PATH/TO/TOKENIZER \
  --shuffle
```

`--max_prompt_tokens` is the prompt length filter threshold.

`--tokenizer_path` must point to the tokenizer used for prompt filtering.

#### SFT Training

Train from the merged CSV

Example:

```bash
torchrun --standalone --nproc_per_node=4 --master_port=47086 \
  -m AnomaMind.train.sft \
  --model_path /PATH/TO/BASE_MODEL \
  --train_csv data/cross_domain/sft.csv \
  --output_dir /PATH/AnomaMind/models/SFT \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --gradient_checkpointing \
  --fsdp
```
### 5. RL Training
#### Build RL Parquet

Convert the merged SFT CSV into RL parquet files.
Use the same tokenizer and prompt-length threshold here as in SFT CSV preparation:

```bash
python -m AnomaMind.train.prepare_rl_data \
  --mode csv \
  --csv data/cross_domain/sft.csv \
  --out-dir data/cross_domain/RL \
  --data-source anomamind/cross_domain \
  --ability ts_anomaly_detection \
  --max-prompt-tokens 7000 \
  --tokenizer-path /PATH/TO/TOKENIZER \
  --val-ratio 0
```


#### Training Command

Run RL with the parquet data and the SFT model.

Example:

```bash
export TRAIN_PARQUET=/PATH/AnomaMind/data/cross_domain/RL/train.parquet
export VAL_PARQUET=/PATH/AnomaMind/data/cross_domain/RL/val.parquet
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/PATH/AnomaMind/models/SFT
export CKPT_DIR=/PATH/AnomaMind/models/RL

bash scripts/run_verl.sh
```
### 6. Evaluation

#### Start the RL model with vLLM

```bash
ASCEND_DEVICES=0,1,2,3 MODEL_PATH=/PATH/AnomaMind/models/RL bash scripts/serve_vllm.sh
```

#### Point `env/detector.env` to the local server

Before running evaluation inference, replace `env/detector.env` with:

```env
LLM_API_KEY=dummy
LLM_BASE_URL=http://localhost:8098/v1
AUX_LLM_MODEL=detector
```

#### Run inference on each test set

```bash
python -m AnomaMind.infer.infer \
  data/<name>/processed/test \
  --output-dir data/<name>/result/test \
  --workers 8 \
  --max-retries 3
```

#### Evaluate the results

Run the evaluation script on the result directory:

```bash
python -m AnomaMind.eval.evaluate_results data/<name>/result/test
```




## 📊 Benchmark Results

**Main Results (F1 / Best-F1):**

AnomaMind achieves competitive or best average performance across Precision, Recall, F1, and Best-F1 on YAHOO, TODS, IOPS, and WSD, outperforming statistical (FFT-AD, SR), deep learning (CNN, M2N2, LSTMAD, TransAD), foundation (TimesFM, Chronos), and LLM-based (LLMAD, LLM-TSAD, OFA) baselines.

<p align="center">
  <img src="assets/1-main.png" width="800">
</p>

**Ablation Studies:**

- **Detection tools**: Removing interval localization, feature extraction, or knowledge memory each degrades performance; feature extraction has the largest impact.
- **Workflow**: Removing the evaluator (iterative refinement) consistently hurts F1 and Best-F1 across datasets.
- **Reinforcement learning**: Removing RL for anomaly decision learning leads to substantial performance drop; RL is critical for task-specific decision boundaries.
<p align="center">
  <img src="assets/2-tool.png" width="800">
</p>

<p align="center">
  <img src="assets/3-rl.png" width="45%">
  &emsp;
  <img src="assets/4-agent.png" width="45%">
</p>

---

## 📁 Project Structure

```
AnomaMind/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── data/              # Place benchmark data here
├── assets/            # Figures for README
├── env/               # Local env examples
├── scripts/           # Training / serving scripts
└── src/               # Python package source
```

## 🤖 Model Zoo

We provide pre-trained detector models for different scales:

| Model | Description | Hugging Face Repo |
|-------|-------------|-------------------|
| **AnomaMind-0.6B** | The model based on Qwen3-0.6B | [AnomaMind-0.6B](https://huggingface.co/YuChongZ/AnomaMind-0.6B) |
| **AnomaMind-8B** | The model based on Qwen3-8B | [AnomaMind-8B](https://huggingface.co/YuChongZ/AnomaMind-8B) |

You can download and use these models directly:

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{tao2026anomamind,
  title={AnomaMind: Agentic Time Series Anomaly Detection with Tool-Augmented Reasoning},
  author={Tao, Xiaoyu and Wu, Yuchong and Cheng, Mingyue and Guo, Ze and Gao, Tian},
  journal={arXiv preprint arXiv:2602.13807},
  year={2026}
}
```
