# 📘 MiniGPT – Character-Level Transformer from Scratch

A lightweight implementation of a GPT-style Language Model built from scratch using PyTorch.  
This project demonstrates the core building blocks behind modern Large Language Models (LLMs), including self-attention, transformer layers, and autoregressive text generation.

---

## 🚀 Project Overview

MiniGPT is a simplified implementation of a Generative Pre-trained Transformer (GPT) that learns patterns from text data and generates new text character-by-character.

The goal of this project is to provide a clear and practical understanding of how LLMs work internally, without relying on high-level libraries.

---

## 🔧 Key Features

- Transformer-based architecture (GPT-style)
- Multi-head self-attention mechanism
- Character-level tokenization
- Autoregressive text generation
- Training with loss monitoring
- Modular and clean code structure

---

## 📂 Project Structure

miniGPT/  
│  
├── config.py – Model and training configuration  
├── tokenizer.py – Character-level tokenizer  
├── dataset.py – Batch generation logic  
├── attention.py – Self-attention and multi-head attention  
├── transformer_block.py – Transformer block (attention + feedforward)  
├── model.py – MiniGPT architecture  
├── train.py – Training pipeline  
├── generate.py – Text generation script  
├── input.txt – Training dataset  
└── README.md – Project documentation  

---

## 🧠 Model Architecture

The model follows a simplified GPT pipeline:

Input Tokens → Embeddings → Transformer Blocks → Layer Normalization → Linear Layer → Output Probabilities

### Core Components

- Token Embeddings: Converts characters into dense vector representations  
- Positional Embeddings: Adds information about token positions in the sequence  
- Multi-Head Attention: Captures relationships between tokens in context  
- FeedForward Network: Applies non-linear transformations  
- Layer Normalization: Stabilizes and improves training  
- Linear Head: Predicts the next token probability  

---

## ⚙️ Configuration

The model configuration is defined in the configuration file and includes parameters such as:

- Context window size  
- Embedding dimension  
- Number of attention heads  
- Number of transformer layers  
- Dropout rate  
- Learning rate  
- Training iterations  
- Device selection (CPU or GPU)  

The vocabulary size is dynamically determined based on the dataset.

---

## 📊 Training Details

### Dataset

- Input text file used for training  
- Approximately 232K characters  
- Around 81 unique characters  

### Training Behavior

- Training loss decreases steadily over iterations  
- Validation loss closely follows training loss  
- Indicates that the model is learning meaningful patterns without heavy overfitting  

---

## ✍️ Text Generation

The model generates text by predicting one character at a time based on previous context.

### Example Prompt

The boy laughed cheerfully and

### Example Output

and stight a beat wing molde the way boy were a plach oght the could the was of the he have nead we...

---

## 🤔 Why the Output Looks Imperfect

The generated text may appear partially correct but not fully coherent. This is expected due to:

- Character-level modeling, which makes learning full words and grammar more difficult  
- Small model size, limiting its learning capacity  
- Limited training iterations  
- Probabilistic sampling, which introduces randomness  

---

## ✅ What the Model Learns Successfully

- Basic English structure  
- Word boundaries  
- Frequently occurring patterns  
- Sentence-like flow  

---

## ❌ Current Limitations

- Spelling inconsistencies  
- Incomplete or broken words  
- Weak long-range context understanding  
- Occasional repetition  

---

## 🛠️ Potential Improvements

- Increase training duration  
- Improve sampling strategies (temperature, top-k, top-p)  
- Use more advanced tokenization methods (word-level or subword/BPE)  
- Increase model size and depth  
- Train on a larger and more diverse dataset  

---

## ▶️ How to Run

1. Install required dependencies  
2. Train the model using the training script  
3. Generate text using the generation script  

---

## 📌 Key Learnings

- Understanding transformer architecture from scratch  
- Importance of tokenizer and model alignment  
- Difference between memorization and generation  
- Role of sampling in text generation  
- Debugging real-world deep learning issues  

---

## 🎯 Future Scope

- Implement advanced sampling techniques  
- Build an interactive interface for text generation  
- Upgrade to word-level or subword tokenization  
- Deploy as an API or web application  

---

## 🙌 Conclusion

This project serves as a strong foundation for understanding how modern language models work internally.  
While the generated text is not perfect, it demonstrates the ability of transformer models to learn language patterns and generate new sequences.
