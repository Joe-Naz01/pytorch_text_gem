# Text Generation Demos (PyTorch + HF Transformers)

This notebook bundles four focused text-generation experiments:

1. **Char-RNN from scratch (PyTorch)**  
   - Custom `RNNmodel` using `nn.RNN` → `nn.Linear`  
   - Trains on a short text snippet (e.g., Alice-like sample)  
   - Next-character prediction with cross-entropy

2. **Toy Character GAN**  
   - `Generator` and `Discriminator` are small `nn.Sequential` MLPs  
   - Operate on the character vocabulary vector  
   - Prints a few generated characters (illustrative, not SOTA)

3. **GPT-2 Free-form Sampling**  
   - `GPT2Tokenizer`, `GPT2LMHeadModel`  
   - Generate with controls like `temperature`, `top_k`, `top_p`,
     `no_repeat_ngram_size`, `max_length`

4. **T5-Small Translation**  
   - `T5Tokenizer`, `T5ForConditionalGeneration`  
   - Example prompt: `translate English to French: 'Hello, ...'`

5. **BLEU & ROUGE with torchmetrics**  
   - Quick, reference vs generated comparison to sanity-check outputs

---

## Quickstart

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Launch
jupyter lab  # or: jupyter notebook

4) Git clone
git clone https://github.com/Joe-Naz01/pytorch_text_gem.git
cd pytorch_text_gem
