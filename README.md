# Multimodal Planning Framework

This repository includes code and materials for the ACL2025-Findings paper "Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation".

## 🚀 Features

- **Multiple AI Backends**: Currently support for GPT-4o, Gemini, and Mistral models
- **Various Processing Modes**: From simple text-based to complex visual-enhanced planning
- **Modular Architecture**: Easy to extend with new processors and backbones

## 📁 Project Structure

```
├── planner.py           # Main entry point
├── config.py            # Configuration management
├── models.py            # Model backends and management
├── processors.py        # Task processing implementations
├── pipeline.py          # Main processing pipeline
├── utils.py             # Utility functions and helpers
└── README.md            # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd llm_planning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   ```

## 🚀 Usage

### Basic Usage

```bash
python planner.py --mode vanilla --backbone gpt4o --model gpt-4o --save_dir ./results
```

### Advanced Usage

```bash
python planner.py \
  --mode tip \
  --backbone gem \
  --model gemini-1.5-flash \
  --data_dir ./dataset/tasks.csv \
  --save_dir ./results \
  --start_idx 0 \
  --end_idx 50 \
  --temperature 0.2 \
  --seed 42
```

### Available Commands

- **List available modes**:
  ```bash
  python planner.py --list_modes
  ```

- **Validate configuration**:
  ```bash
  python planner.py --validate_config --backbone gpt4o --model gpt-4o --save_dir ./test
  ```

- **Dry run** (show what would be processed):
  ```bash
  python planner.py --dry_run --mode vanilla --backbone gpt4o --save_dir ./results
  ```

## 🎯 Processing Modes

| Mode | Description |
|------|-------------|
| `vanilla` | Generate textual plan first, then create visual plan |
| `stable` | Generate visual plan first, then create texual plan |
| `tip` | TIP-based image generation and revision |
| `w_des` | Textual plan refinement with detailed image descriptions |
| `w_img` | Textual plan refinement with visual interpretation |
| `ours` | Ours approach with pPDDL visual information and coherent image generation |

## 🤖 Supported Backbones

### GPT-4O
- Models: `gpt-4o`, `gpt-4o-mini`
- Requires: `OPENAI_API_KEY`

### Gemini
- Models: `gemini-1.5-flash`
- Requires: `GOOGLE_API_KEY`

### Mistral
- Models: `mistral-7b`, `mistral-8x7b`

## 📊 Output Structure

Each processed task creates a directory with the following structure:

```
results/
└── task_0/
    ├── ori_plan.txt      # Original generated plan
    ├── rev_plan.txt      # Revised plan (if applicable)
    ├── descriptions.txt  # Image descriptions
    ├── captions.txt      # Image captions
    ├── step_1.png        # Generated images
    ├── step_2.png
    └── ...
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.