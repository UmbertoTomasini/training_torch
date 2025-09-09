# PyTorch Training Repository

This repository contains various PyTorch training exercises, implementations, and experiments.

## Project Structure

```
pytorch_training/
├── basics/                    # Basic PyTorch concepts and implementations
│   ├── tensors.py            # Tensor operations and basics
│   ├── build_nn.py           # Neural network construction
│   ├── optim.py              # Optimization techniques
│   ├── datasets_dataloaders.py # Data handling and loaders
│   ├── transform.py          # Data transformations
│   └── data/                 # Sample datasets
├── exercises/                 # Training exercises and experiments
│   ├── train_loop.py         # Training loop implementation
│   ├── transfer_learning.py  # Transfer learning examples
│   ├── model_self_att.py     # Self-attention model implementation
│   ├── model_mlp_classifier.py # MLP classifier
│   ├── custom_loss.py        # Custom loss functions
│   ├── forward_hook.py       # Forward hooks
│   ├── hook_attn.py          # Attention hooks
│   ├── inference_autoregr.py # Autoregressive inference
│   ├── inference_class.py    # Classification inference
│   ├── shape_broadcast.py    # Shape and broadcasting examples
│   ├── data_nlp_ex1.py       # NLP data processing example 1
│   ├── data_nlp_ex2.py       # NLP data processing example 2
│   └── llama2/               # LLaMA2 related implementations
└── torch_interview/          # Interview preparation materials
    └── test.py               # Test implementations
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Other dependencies as needed for specific exercises

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd pytorch_training
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib
# Add other dependencies as needed
```

## Usage

Each script can be run independently. For example:

```bash
# Run basic tensor operations
python basics/tensors.py

# Run neural network construction
python basics/build_nn.py

# Run training loop
python exercises/train_loop.py
```

## Contents Overview

### Basics
- **tensors.py**: Introduction to PyTorch tensors, operations, and basic manipulations
- **build_nn.py**: Building neural networks using PyTorch's nn module
- **optim.py**: Optimization techniques and custom optimizers
- **datasets_dataloaders.py**: Data loading and preprocessing
- **transform.py**: Data transformation utilities

### Exercises
- **train_loop.py**: Complete training loop implementation
- **transfer_learning.py**: Transfer learning examples and techniques
- **model_self_att.py**: Self-attention mechanism implementation
- **model_mlp_classifier.py**: Multi-layer perceptron classifier
- **custom_loss.py**: Custom loss function implementations
- **forward_hook.py**: Using forward hooks for model inspection
- **hook_attn.py**: Attention mechanism hooks
- **inference_autoregr.py**: Autoregressive inference implementation
- **inference_class.py**: Classification inference utilities
- **shape_broadcast.py**: Understanding tensor shapes and broadcasting
- **data_nlp_ex1.py** & **data_nlp_ex2.py**: NLP data processing examples

### Interview Preparation
- **torch_interview/**: Materials for PyTorch interview preparation

## Learning Path

1. Start with `basics/` to understand fundamental PyTorch concepts
2. Move to `exercises/` for hands-on training implementations
3. Use `torch_interview/` for interview preparation

## Contributing

Feel free to contribute by:
- Adding new exercises
- Improving existing implementations
- Fixing bugs
- Adding documentation

## License

This project is for educational purposes. Please ensure you have the right to use any datasets or pre-trained models you work with.

## Notes

- Some files may contain large model files or datasets that are not tracked in git
- Check individual files for specific requirements and usage instructions
- This repository is primarily for learning and experimentation purposes