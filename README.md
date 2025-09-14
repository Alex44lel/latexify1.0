# Latexify 1.0 🔢➡️📝

An end-to-end deep learning system for converting mathematical formula images to LaTeX code. This project implements various state-of-the-art vision-language architectures to tackle the challenging task of optical formula recognition.

## 🎯 Overview

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
