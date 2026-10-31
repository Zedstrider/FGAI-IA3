# Machine Translation System Architectures: Encoder-Decoder vs. Decoder-Only

## Problem Statement and Objective
The objective of this project is to develop and evaluate machine translation systems translating English to French. The project compares the performance, computational requirements, and translation quality of traditional encoder-decoder architectures against modern decoder-only Large Language Models (LLMs).

## Dataset Description and Source Link
* **Dataset:** WMT14 (Workshop on Machine Translation 2014) English-French parallel-text dataset.
* **Subset:** To manage computational resources, a 25,000-sample subset was extracted via streaming and divided into training (20,000), validation (2,500), and test (2,500) sets.
* **Source Link:** [Hugging Face: wmt/wmt14](https://huggingface.co/datasets/wmt/wmt14)

## Models Implemented
1. **Basic Transformer (Encoder-Decoder):** Initialized from scratch using the `Helsinki-NLP/opus-mt-en-fr` configuration and trained strictly on the 20,000-sample subset.
2. **mT5-small (Encoder-Decoder):** A 300M parameter pre-trained multilingual model (`google/mt5-small`), fine-tuned for 3 epochs on the dataset using task-specific prefixes.
3. **LLaMA-3.2-1B (Decoder-Only):** A 1-billion parameter model quantized to 4-bit precision, utilizing few-shot prompting to guide the translation structure.
4. **Gemma-2B-IT (Decoder-Only):** A 2-billion parameter instruction-tuned variant quantized to 4-bit precision, utilizing direct zero-shot prompting.

## Execution Steps and Required Dependencies
**Dependencies:**
`pip install datasets transformers[torch] accelerate bitsandbytes evaluate sacrebleu`

**Execution Steps:**
1. Authenticate with Hugging Face using a valid access token.
2. Load and split the dataset using `datasets`.
3. Train the Basic Transformer from scratch (3 epochs, FP16).
4. Fine-tune the mT5-small model (3 epochs, FP32).
5. Load LLaMA-3.2-1B and Gemma-2B-IT in 4-bit precision using `BitsAndBytesConfig`.
6. Run batch inference (batch size 32 for encoder-decoders; sequential generation for decoder-only).
7. Calculate BLEU and chrF scores using the `evaluate` library.

## Model Comparison and Evaluation
| Model | Architecture | Parameters | Quantization | Inference Time (2,500 samples) | BLEU Score | chrF Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Basic Transformer** | Encoder-Decoder | ~74M | FP16 | ~37 seconds | 4.46 | 24.28 |
| **mT5-small** | Encoder-Decoder | ~300M | FP32 | ~1 min 26 seconds | 14.06 | 41.22 |
| **LLaMA-3.2-1B** | Decoder-Only | 1 Billion | 4-bit (NF4) | ~64 mins 24 seconds | 7.82 | 37.95 |
| **Gemma-2B-IT** | Decoder-Only | 2 Billion | 4-bit (NF4) | ~58 mins 34 seconds | **15.99** | **46.96** |

The results demonstrate massive differences in both performance and computational requirements. The traditional encoder-decoder architectures process the entire input sequence simultaneously, resulting in vastly superior parallelized inference speeds (~37 to 86 seconds). Conversely, the autoregressive decoder-only models process prompts and translations token-by-token sequentially, creating a heavy inference bottleneck (~60 hours) while requiring 4-bit quantization to fit into standard GPU memory.

## Sampling-Temperature Analysis & Ethical AI Concerns
During implementation, several responsible AI concerns regarding bias, cultural misrepresentation, and model safety were encountered and mitigated:

* **Hallucinated Information:** Initial tests with the base `Gemma-2B` model resulted in severe hallucinations, where the model completely ignored the translation prompt and fabricated extensive corporate business scenarios instead. 
* **Mitigation via Instruction Tuning:** This ethical concern was mitigated by switching to the `Gemma-2B-IT` (Instruction-Tuned) variant. Instruction tuning securely grounds the model to follow explicit directives ("Translate the following... Do not include explanations"), proving vital for safe, predictable AI deployment.
* **Token Degeneration & Sampling Strategy:** LLaMA-3.2-1B exhibited infinite token generation loops (e.g., repeating "en effet, en effet..."). To ensure reliable translations, probabilistic sampling (temperature adjustments) was explicitly disabled (`do_sample=False`). Switching to deterministic greedy decoding, paired with a `repetition_penalty=1.2`, successfully forced the model to select the most probable tokens without falling into degenerative loops.
* **Bias and Cultural Representation:** The Basic Transformer trained on just 20,000 rows (BLEU 4.46) struggled heavily with context, highlighting that deploying under-trained models risks generating broken, inappropriate, or culturally insensitive translations. Relying on robustly pre-trained multi-lingual foundations (like mT5) significantly reduces these risks.

## Conclusion Identifying the Best-Performing Model
**Gemma-2B-IT** is the best-performing model for translation accuracy, achieving the highest BLEU (15.99) and chrF (46.96) scores. However, when balancing translation quality with computational viability, **mT5-small** is the most practical architecture. Despite being drastically smaller, mT5 achieved highly competitive accuracy (BLEU 14.06) while completing inference 45 times faster than Gemma, demonstrating that fine-tuned encoder-decoder models remain superior for large-scale, dedicated translation workloads.
