# Latexify 1.0 🔢➡️📝

An end-to-end deep learning system for converting mathematical formula images to LaTeX code. This project implements various state-of-the-art vision-language architectures to tackle the challenging task of optical formula recognition.

## 🎯 Overview

Ever struggled with typing complex mathematical equations? This project aims to solve that by automatically converting images of mathematical formulas into LaTeX code. We experimented with different encoder-decoder architectures to find the best combination for accurate formula recognition.

The system uses:
- **Vision Encoders**: ResNet, ConvNeXt, and Swin Transformer variants
- **Language Decoders**: Transformer and GPT-based architectures  
- **Custom Tokenization**: LaTeX-specific tokenization strategies
- **Comprehensive Evaluation**: BLEU scores, visual matching, and syntactic correctness

## 🏗️ Architecture

Our approach follows an encoder-decoder paradigm:

```
Mathematical Formula Image → Vision Encoder → Feature Representation → Language Decoder → LaTeX Sequence
```

### Supported Encoders
- **ResNet**: ResNet-18/34/50/101/152 (11M - 58M parameters)
- **ConvNeXt**: Tiny/Small/Base/Large variants (27M - 196M parameters)  
- **Swin Transformer**: V2 Tiny/Small/Base (27M - 86M parameters)

### Supported Decoders
- **Transformer**: Classic encoder-decoder architecture
- **GPT**: Decoder-only generative approach
- **LSTM**: With soft attention mechanism (experimental)

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Preparation
We use a dataset of mathematical formula images paired with their LaTeX representations:

```bash
# Download and prepare the dataset
sh ./scripts/get_data.sh
```

### Training a Model
You can train different model combinations easily:

```python
from src.train import train, get_model

# Example: ResNet-50 + GPT decoder
model, tokenizer = get_model("resnet50", "gpt", {}, {}, {})

# Load your data loaders (see preprocessing.ipynb for details)
train_loader = get_data_loaders(df_train, labels_train, "train", tokenizer, max_length)
test_loader = get_data_loaders(df_test, labels_test, "test", tokenizer, max_length)

# Start training
train(train_loader, test_loader, model, tokenizer, num_epochs=30)
```

### Generate LaTeX from Images
```python
from src.generate import generate
from PIL import Image

# Load your trained model
model.load_state_dict(torch.load("./models/latexify-resnet50-gpt.pth"))
model.eval()

# Generate LaTeX for a new image
image = Image.open("math_formula.png")
latex_code = generate(model, image, tokenizer, max_new_tokens=200)
print(f"Generated LaTeX: {latex_code}")
```

## 📊 Evaluation Metrics

We evaluate our models using multiple metrics to ensure both accuracy and practical utility:

1. **Corpus BLEU (4-gram)**: Measures n-gram overlap with reference sequences
2. **Visual Match Score**: Exact visual matching between rendered formulas
3. **Image Edit Distance Accuracy (EDA)**: Captures visual similarity
4. **Syntactic Correctness**: Percentage of valid LaTeX sequences

## 🔬 Experimental Results

Our experiments show that:
- **ConvNeXt + Transformer** achieves the best balance of accuracy and efficiency
- **Swin Transformer** excels on complex, multi-line formulas
- **GPT decoders** show better generalization to unseen formula patterns
- Performance degrades gracefully with formula length (consistent with literature)

See `analysis/` directory for detailed performance analysis and visualizations.

## 📁 Project Structure

```
├── src/                    # Core implementation
│   ├── encoders/          # Vision encoder architectures
│   ├── decoders/          # Language decoder architectures
│   ├── models.py          # Main LatexifyModel class
│   ├── tokenizer.py       # LaTeX tokenization
│   ├── train.py           # Training pipeline
│   └── generate.py        # Inference utilities
├── tests/                 # Unit tests
├── analysis/              # Performance analysis notebooks
├── docs/                  # Documentation and goals
├── scripts/               # Utility scripts
├── main.ipynb            # Main training notebook
└── preprocessing.ipynb    # Data analysis and preprocessing
```

## 💻 Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
We follow standard Python conventions with black formatting:
```bash
black src/ tests/
```

### Notebooks
- `main.ipynb`: Complete training pipeline with examples
- `preprocessing.ipynb`: Data exploration and preprocessing steps

## 🎓 Research Context

This project builds upon recent advances in vision-language models and mathematical OCR:

- Inspired by im2markup approaches for formula recognition
- Incorporates attention mechanisms for better sequence modeling
- Evaluates both traditional CNN and modern transformer-based encoders
- Implements beam search and sampling strategies for diverse outputs

## 🤝 Contributing

We welcome contributions! Areas where help would be appreciated:
- Additional encoder architectures (EfficientNet, ViT variants)
- Character-level tokenization experiments
- Multi-modal attention visualizations
- Performance optimizations for mobile deployment

## 📝 Citation

If you use this work in your research, please cite:
```bibtex
@misc{latexify2024,
  title={Latexify 1.0: Deep Learning for Mathematical Formula Recognition},
  author={Alex44lel},
  year={2024},
  url={https://github.com/Alex44lel/latexify1.0}
}
```

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Thanks to the creators of the mathematical formula datasets
- Inspired by the Harvard NLP annotated transformer tutorial
- Built on PyTorch and timm libraries
- Special appreciation to the computer vision and NLP communities

---

*"Mathematics is the language with which God has written the universe."* - Galileo Galilei

Happy formula recognizing! 🧮✨