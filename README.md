# LLM-ODE: Suitability of LLM and Expert Informed Frameworks for Inferring Longitudinal Rare Disease Models

## Overview
**LLM-ODE:** A privacy-preserving framework for discovering ODE-based disease progression models from summary statistics using large language models, through iterative refinement. The framework supports:
- **ODE skeleton discovery** - LLM-guided generation and refinement of ODE system structures informed by domain knowledge
- **Parameter distribution inference** - Estimation of population-level parameter distributions from summary statistics
- **Expert-informed priors** - Incorporation of clinical knowledge through prompt specifications

## Installation

Create a conda environment and install dependencies:

```bash
conda create -n llmode 
conda activate llmode
pip install -r requirements.txt
```

## Dataset
Acute Kidney Injury (AKI) example data extracted from MIMIC-IV 3.1 is provided in [data/aki/](./data/aki). The dataset includes longitudinal clinical measurements for AKI progression.

## Usage

### 1. Start Local LLM Engines

For the **full pipeline** you should have **two local LLM engines running at the same time**:
- Port `5000`: ODE skeleton discovery engine 
- Port `5001`: Parameter distribution inference engine 

To start two engines with different local models manually:

**ODE skeleton discovery:**
```bash
python ./llm_engine/engine.py \
  --model_path "$MODEL_PATH" \
  --gpu_ids [GPU_ID]  \
  --port 5000 \
  --quantization
```

**Paremeter distribution inference:**
```bash
python ./llm_engine/engine.py \
  --model_path "$MODEL_PATH" \
  --gpu_ids [GPU_ID]  \
  --port 5001 \
  --quantization
```
### 2. Run ODE Discovery

```bash
python main.py \
  --problem_name aki \
  --spec_path specs_skeleton/specification_aki_numpy.txt \
  --log_path logs/aki_run1
```

### 3. API-Based Runs

Use OpenAI models instead of local LLMs:
```bash
export API_KEY=your_api_key

python main.py --use_api True \
  --api_model "gpt-4o" \
  --problem_name aki \
  --spec_path specs_skeleton/specification_aki_numpy.txt \
  --log_path logs/aki_api
```

### Configuration
Adjust pipeline parameters in [config.py](llmode/config.py).



## Citation
**This project adapts methods from:**
```bibtex
@article{shojaee2024llm,
  title={Llm-sr: Scientific equation discovery via programming with large language models},
  author={Shojaee, Parshin and Meidani, Kazem and Gupta, Shashank and Farimani, Amir Barati and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2404.18400},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for full details.

This work is adapted from [LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR) by Shojaee et al., which itself builds upon [FunSearch](https://github.com/google-deepmind/funsearch), [PySR](https://github.com/MilesCranmer/PySR), and [Neural Symbolic Regression that scales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales).

We thank the original contributors of these works for open-sourcing their valuable code.
