---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/chuongg3/NLU-EvidenceDetection

---

# Model Card for h25471ds-m19364tg-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a binary classification model that combines ModernBERT and SBERT 
      embeddings to detect whether a piece of evidence supports a given claim (evidence detection). This is a deep learning underpinned by transformer architecture approach.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model uses a dual embedding approach that combines contextualized 
      embeddings from ModernBERT-base with sentence embeddings from SBERT (all-MiniLM-L6-v2). 
      The model first processes claim-evidence pairs through both embedding models, then concatenates 
      the embeddings and passes them through a classifier to predict whether the evidence supports the claim.
      The model is fine-tuned using QLoRA (Quantized Low-Rank Adaptation) with 4-bit quantization 
      and flash-attention for efficient training and inference.

      Text preprocessing includes removing reference tags, normalizing accented characters using unidecode,
      cleaning up irregular spacing around punctuation, and normalizing whitespace. Data augmentation
      was applied to the positive class (minority) using synonym replacement to address class imbalance.

- **Developed by:** Dhruv Sharma and Tuan Chuong Goh
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Dual Embedding Model (ModernBERT + SBERT) with QLoRA fine-tuning
- **Finetuned from model [optional]:** ModernBERT-base and SBERT (all-MiniLM-L6-v2)

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/answerdotai/ModernBERT-base
- **Paper or documentation:** https://huggingface.co/answerdotai/ModernBERT-base

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

Training data consisting of claim-evidence pairs for evidence detection task. Data augmentation was applied to the positive class (minority) using synonym replacement to address class imbalance.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - Fine-tuning method: QLoRA (Quantized Low-Rank Adaptation)
      - Quantization: 4-bit (nf4)
      - Optimization: Uses flash-attention
      - learning_rate: 0.0002643238333834569
      - batch_size: 64
      - num_epochs: 5
      - weight_decay: 0.048207625326781293
      - warmup_ratio: 0.19552784843595056
      - gradient_accumulation_steps: 4
      - lora_r: 56
      - lora_alpha: 40
      - lora_dropout: 0.07644825534662132
      - classifier_dropout: 0.2659719581055393
      - classifier_hidden_size: 768
      - max_length: 8192

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - Model size: The base ModernBERT model is loaded in 4-bit quantization
      - SBERT embeddings dimension: 384
      - Reduced memory footprint due to 4-bit quantization and parameter-efficient fine-tuning

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Development set with claim-evidence pairs for evidence detection.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy: 0.87377657779278
      - Macro Precision: 0.83764094620994
      - Macro Recall: 0.86135532021442
      - Macro F1-Score: 0.84790707217937
      - Weighted Precision: 0.88028808321627
      - Weighted Recall: 0.87377657779278
      - Weighted F1-Score: 0.87591472842040
      - Matthews Correlation Coefficient: 0.69859387983347

### Results

The model achieved a Macro F1-score of 0.848 (84.8%) and an accuracy of 0.874 (87.4%) on the development set.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB
      - GPU: CUDA-compatible GPU with T4 (Turing) architecture or newer
      - Training requires T4 or newer GPU architecture to support flash-attention
      - Inference can be performed on less powerful GPUs with 4-bit quantization

### Software


      - torch 2.6.0+cu126
      - transformers
      - peft (for QLoRA implementation)
      - bitsandbytes (for 4-bit quantization)
      - flash-attn (for efficient attention computation)
      - sentence-transformers
      - sklearn
      - numpy
      - pandas
      - unidecode (for text normalization)
      - re (for text cleaning)

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model uses an optimal threshold of 0.5433 determined through 
      validation data to convert probabilities to binary predictions. The 4-bit quantization may introduce 
      some precision loss compared to full-precision models, although the performance metrics indicate 
      this has minimal impact on model quality. The original dataset had class imbalance which was addressed
      through data augmentation for the positive class.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The model combines the strengths of ModernBERT's long context 
      understanding with SBERT's semantic similarity capabilities. The use of QLoRA and 4-bit quantization 
      enables efficient fine-tuning with significantly reduced memory requirements compared to full-precision 
      fine-tuning. Flash-attention provides computational speedups during training and inference on 
      compatible hardware. 

      Text preprocessing includes:
      1. Removing reference tags like [REF], [REF, REF]
      2. Normalizing accented characters using unidecode
      3. Cleaning up irregular spacing around punctuation
      4. Normalizing whitespace

      The model first extracts the [CLS] token embedding from ModernBERT, then 
      concatenates it with SBERT embeddings before passing through the classification layers. The SBERT embeddings are extracted separately for Claim and Evidence and the combined by averaging before concatenation.

      Hyperparameters were optimized using a systematic search process to find the optimal configuration.

      Important references:

      - QLoRA: Efficient Finetuning of Quantized LLMs (2023) - https://arxiv.org/abs/2305.14314
      - Hugging Face 4-bit Transformers with bitsandbytes - https://huggingface.co/blog/4bit-transformers-bitsandbytes
      - PEFT: Parameter-Efficient Fine-Tuning Documentation - https://huggingface.co/docs/peft/en/index
