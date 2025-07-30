
# **Artha-1**: A Compact Liquid-Autoencoder Language Model

**Huggingface**: [`vyomie/artha-1`](https://huggingface.co/vyomie/artha-1)  
**Model Type**: Hybrid LLM with Liquid Neural Network Core  
**Architecture**: Custom Autoencoder + Liquid Neural Network  
**Model Size**: ~400M parameters  
**Format**: PyTorch `.pth` + Python plug-and-play pipeline  
**Usage**: Plug-and-play via `from model import Pipeline`

[![Watch the video](https://d11a6trkgmumsb.cloudfront.net/original/4X/f/d/d/fddca474f1cf42bdf1c516a8f2e317eadfa7817f.jpeg)](https://youtu.be/aLDtFw47XXA)

---

## 🌍 World's First & Most Affordable Custom LLM – Built by a Teen
---
### **Artha-1** is the **world's first Liquid Neural Network-powered language model pipeline** trained and deployed end-to-end by an independent teen researcher. Unlike traditional LLMs that require millions of dollars, proprietary infrastructure, and industrial compute clusters, **Artha-1 was built in under 3 days using accessible tools and mid-range hardware.**
---
### 💡 What Makes It a World First?

- 🧠 **LNN-Based Reasoning Core:**  
  This is the **first open-source LLM** to integrate a **Liquid Neural Network (LNN)** core for deep, dynamic reasoning inside compressed latent space.

- 💸 **Built With ~Zero Budget:**  
  Trained using just local GPUs and open datasets, **no enterprise backing or funding** was involved. This makes Artha-1 arguably the **cheapest working LLM** architecture available to the public.

- 🧑‍🎓 **Created by a Teen Researcher:**  
  From architecture design and training to deployment and packaging, every step was executed by an **independent teen developer**, proving that **you don’t need a PhD or billion-dollar lab** to innovate in AI.

---

## 📌 Summary

**Artha-1** is a compact and efficient language model designed with an unconventional architecture combining a pretrained autoencoder with a Liquid Neural Network (LNN) core. The model emphasizes interpretability, small footprint, and ease of use for experimentation and lightweight reasoning tasks.
Built with simplicity and modularity in mind, Artha-1 is ideal for research, tinkering, or educational use, and runs efficiently on consumer-grade hardware.
---



---

## 🧪 Architecture

- **Encoder**: Bottleneck-T5 autoencoder (`thesephist/contra-bottleneck-t5-base-wikipedia`)
- **Core Processor**: Liquid Neural Network (LNN) with dynamic temporal memory
- **Decoder**: Same T5 decoder via latent perturbation and reconstruction
- **Interface**: Python `Pipeline` class (plug-and-play)

---

## 🧠 Intended Use

- Lightweight reasoning tasks  
- Prompt-based experimentation  
- Research on alternative LLM architectures  
- Educational demos for architecture breakdowns  
- Fine-tuning or distillation experiments for compact models  

## 🚫 Not Intended Use

- Do not deploy in high-stakes environments (medical, legal, safety-critical tasks)  
- Not optimized for factual correctness or robustness  
- Not meant to replace larger foundational models (GPT, LLaMA, Claude, etc.)  

## 📦 Model Architecture

This model is a two-part system:
- A **Bottleneck T5 autoencoder** for text-to-latent and latent-to-text conversion, adapted from [`thesephist/contra-bottleneck-t5-base-wikipedia`](https://huggingface.co/thesephist/contra-bottleneck-t5-base-wikipedia)
- A custom **Liquid Neural Network (LNN)** core trained to perform latent-level reasoning on compressed embeddings.

The LNN consists of multiple gated recurrent layers designed for temporal and structural memory propagation, allowing for highly expressive representations at low parameter count.

## 📊 Training Details

- **Training Data:** Synthetic question-answer dataset generated using open-source LLMs  
- **Latent Size:** 768  
- **LNN Hidden Units:** 4000  
- **Training Duration:** ~2–3 days on mid-range GPUs  
- **Optimizer:** AdamW with SWA (Stochastic Weight Averaging)  
- **Loss Function:** Cosine similarity between predicted and true bottleneck embeddings  

## 🛠️ How to Use

Make sure you have the following dependencies installed:

```bash
pip install torch transformers==4.36.1 huggingface_hub
```

Import required packages

```bash
import os
import sys
import importlib.util
from huggingface_hub import snapshot_download
```
Now, download custom Pipeline dynamically

```bash
snapshot_download("vyomie/artha-1", local_dir="/tmp/vyomie_artha-1", local_dir_use_symlinks=False)

spec = importlib.util.spec_from_file_location("model", "/tmp/vyomie_artha-1/model.py")
model = importlib.util.module_from_spec(spec)
sys.modules["model"] = model
spec.loader.exec_module(model)
```

Initialize the Pipeline

```bash
Pipe = Pipeline("vyomie/artha-1")
```

```bash
Output = Pipe("Hi, how are you!")
print(Output)
```
