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
python planner.py --mode passage --backbone gpt4o --model gpt-4o --save_dir ./results
```

### Advanced Usage

```bash
python planner.py \
  --mode step_tip \
  --backbone gem \
  --model gemini-pro \
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
  python planner.py --dry_run --mode passage --backbone gpt4o --save_dir ./results
  ```

## 🎯 Processing Modes

| Mode | Description |
|------|-------------|
| `passage` | Generate complete plan first, then create images |
| `step` | Generate plan step-by-step interactively |
| `passage_tip` | Passage mode with tip-based image generation |
| `step_tip` | Step mode with tip-based enhancement |
| `passage_w_des` | Passage mode with detailed image descriptions |
| `step_w_des` | Step mode with detailed descriptions |
| `pure_img` | Generate only images for steps |
| `img_2_text` | Convert existing images to text descriptions |

## 🤖 Supported Backbones

### GPT-4O
- Models: `gpt-4o`, `gpt-4o-mini`
- Requires: `OPENAI_API_KEY`

### Gemini
- Models: `gemini-pro`, `gemini-pro-vision`
- Requires: `GOOGLE_API_KEY`

### Mistral
- Models: `mistral-7b`, `mistral-8x7b`
- Status: Implementation in progress

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

## 🔧 Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--backbone` | Model backbone (gpt4o/gem/mistral) | `gpt4o` |
| `--model` | Specific model name | `gpt-4o` |
| `--mode` | Processing mode | Required |
| `--temperature` | Sampling temperature | `0.1` |
| `--seed` | Random seed | `42` |
| `--start_idx` | Starting task index | `0` |
| `--end_idx` | Ending task index | `100` |
| `--data_dir` | Dataset file path | `./dataset/wikiHow_tasks_merge.csv` |
| `--save_dir` | Output directory | Required |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models |
| `GOOGLE_API_KEY` | Google API key for Gemini models |
| `CUDA_VISIBLE_DEVICES` | GPU device selection |

## 🔍 Validation and Testing

The system includes comprehensive validation:

1. **Configuration Validation**:
   - Model compatibility checking
   - File path validation
   - Parameter range validation

2. **Connectivity Testing**:
   - API key validation
   - Model response testing
   - Dataset accessibility

3. **Runtime Monitoring**:
   - Progress tracking
   - Error logging
   - Success rate reporting


## 🚧 Extending the System

### Adding New Processors

1. Create a new processor class in `processors.py`:
   ```python
   class CustomProcessor(TaskProcessor):
       def _process_task_impl(self, task: str, task_path: str) -> bool:
           # Your implementation here
           return True
   ```

2. Register it in `pipeline.py`:
   ```python
   processors['custom_mode'] = CustomProcessor(self.model_manager)
   ```

### Adding New Backbones

1. Implement the backbone interface in `models.py`:
   ```python
   class CustomBackbone(BackboneInterface):
       def get_text_response(self, prompt, seed, temperature):
           # Your implementation
           pass
   ```

2. Register it in the backbone map and update configuration.

## 📈 Performance Tips

1. **Batch Processing**: Process in smaller batches for better memory management
2. **GPU Memory**: Monitor CUDA memory usage with large models
3. **API Limits**: Be aware of rate limits for external APIs
4. **Storage**: Ensure sufficient disk space for generated images

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review error logs in the output
- Submit an issue with detailed information