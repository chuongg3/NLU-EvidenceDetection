# NLU-Evidence Detection

This repository contains implementations of two different approaches for Natural Language Understanding (NLU) Evidence Detection:
1. Co-Attention Siamese Deep Learning Model
2. ModernBERT SBERT Dual Embedding Model

## Repository Structure

```
NLU-EvidenceDetection/
├── training_data/                           # Training data for all models
├── CoAttentionSiameseDeepLearning.ipynb     # Notebook for Siamese model training
├── CoAttentionSiameseDeepLearning.keras     # Saved Siamese model
├── CoAttentionSiameseEvaluation.ipynb       # Evaluation notebook for Siamese model
├── CoAttentionSiameseInference.ipynb        # Inference notebook for Siamese model
├── CoAttentionSiamese_model_card.md         # Model card with details for Siamese model
├── Group_7_B.csv                            # Test data predictions for deep learning model
├── Group_7_C.csv                            # Updated test files
├── README.md                                # This file
├── modernbert_sbert_dual_embedding_model_card.md    # Model card for BERT model
├── modernbert_sbert_embeddings.ipynb        # BERT embeddings training notebook
├── modernbert_sbert_embeddings_evaluation.ipynb     # BERT model evaluation
└── modernbert_sbert_embeddings_inference.ipynb      # BERT model inference
```

## Model 1: Co-Attention Siamese Deep Learning Model

### Overview
This model utilizes a Siamese neural network architecture with co-attention mechanisms to detect evidence in text. The model takes pairs of text (claim and potential evidence) and predicts whether the second text provides evidence for the first.

### Model Architecture
- **Siamese Network**: Dual-path network that processes two texts separately before comparing them
- **Multi-Headed Co-Attention Mechanism**: Allows the claims to attend to the evidence, and vice versa
- **Embedding Process**: 
  - Text inputs first pre-processed by removing '[ref]', '[ref' and 'ref]'
  - Every text encoded by Sentence BERT into a 384 dimensional vector
  - Siamese encoder shared by both claim and evidence embeddings to produce task-relevant embeddings

### How to Run

1. **Setup Environment**:
   ```bash
   pip install tensorflow keras numpy pandas scikit-learn matplotlib
   ```

2. **Training**:
   - Open `CoAttentionSiameseDeepLearning.ipynb` in Jupyter Notebook or Google Colab
   - Follow the instructions to load training data and train the model
   - The trained model will be saved as `.keras` file

3. **Evaluation**:
   - Use `CoAttentionSiameseEvaluation.ipynb` to evaluate model performance
   - The notebook includes confusion matrix visualization and performance metrics

4. **Inference**:
   - Use `CoAttentionSiameseInference.ipynb` for making predictions on new data

### Performance
The model achieved the following metrics on the development set:
- Accuracy: 84.48%
- Macro F1-Score: 80.12%
- Macro Precision: 80.97%
- Macro Recall: 79.40%

The model was trained for 23 epochs with optimized hyperparameters determined through Bayesian optimization.

## Model 2: ModernBERT SBERT Dual Embedding Model

### Overview
This model leverages pre-trained BERT models (specifically Sentence-BERT) to create embeddings for text pairs and then determines evidence relationships using these embeddings.

### Model Architecture
- **Dual Embedding Approach**: Combines contextualized embeddings from ModernBERT-base with sentence embeddings from SBERT (all-MiniLM-L6-v2)
- **Text Processing**: 
  - Removes reference tags
  - Normalizes accented characters using unidecode
  - Cleans irregular spacing around punctuation
  - Normalizes whitespace
- **Training Approach**: Fine-tuned using QLoRA (Quantized Low-Rank Adaptation) with 4-bit quantization and flash-attention for efficiency

### How to Run

1. **Setup Environment**:
   ```bash
   pip install transformers sentence-transformers torch pandas numpy scikit-learn matplotlib
   ```

2. **Training/Fine-tuning**:
   - Open `modernbert_sbert_embeddings.ipynb` in Jupyter Notebook or Google Colab
   - Follow the notebook to load data and fine-tune the model

3. **Evaluation**:
   - Use `modernbert_sbert_embeddings_evaluation.ipynb` to evaluate model performance

4. **Inference**:
   - Use `modernbert_sbert_embeddings_inference.ipynb` for making predictions on new data

### Performance
The model achieved the following metrics on the development set:
- Accuracy: 87.38%
- Macro F1-Score: 84.79%
- Macro Precision: 83.76%
- Macro Recall: 86.14%
- Matthews Correlation Coefficient: 69.86%

The model uses an optimal threshold of 0.5433 determined through validation data to convert probabilities to binary predictions.

## Data Sources and Attribution

### Training Data
- The training data is located in the `training_data/` directory
- The dataset consists of 30K pairs of texts drawn from emails, news articles, and blog posts
- For the Siamese model, 21K pairs were used for training and 6K for validation
- For the ModernBERT model, data augmentation was applied to the positive class (minority) using synonym replacement to address class imbalance

### Pre-trained Models
- Both models utilize Sentence-BERT embeddings from `all-MiniLM-L6-v2`
- The ModernBERT implementation also uses the [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) model from HuggingFace

### Cloud-stored Models
- Full trained models are stored on OneDrive due to size constraints:
  - Siamese Model: [OneDrive Link](https://onedrive.com/link/to/siamese/model)
  - BERT Model: [OneDrive Link](https://onedrive.com/link/to/bert/model)
  - *Note: Replace with actual OneDrive links*

## Citation

If you use this code or models in your research, please cite:

```
@misc{NLU-EvidenceDetection,
  author = {Tuan Chuong Goh and Dhruv Sharma},
  title = {NLU-Evidence Detection},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/chuongg3/NLU-EvidenceDetection}}
}
```

For the Co-Attention mechanism implementation, please also cite:
```
@inproceedings{lu2016hierarchical,
  title={Hierarchical question-image co-attention for visual question answering},
  author={Lu, Jiasen and Yang, Jianwei and Batra, Dhruv and Parikh, Devi},
  booktitle={Advances in neural information processing systems},
  pages={289--297},
  year={2016},
  url={https://proceedings.neurips.cc/paper_files/paper/2016/file/9dcb88e0137649590b755372b040afad-Paper.pdf}
}
```

For the ModernBERT model, please cite:
```
@misc{modernbert,
  author = {Answer.ai},
  title = {ModernBERT},
  year = {2023},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/answerdotai/ModernBERT-base}}
}
```

## License
This project is licensed under the CC-BY-4.0 License - see the LICENSE file for details.

## Acknowledgments
- The Sentence-BERT team for their pre-trained models
- Answer.ai for the ModernBERT-base model
- The TensorFlow and PyTorch teams for their frameworks
- Jiasen Lu et al. for their work on hierarchical co-attention mechanisms
- The PEFT, bitsandbytes, and flash-attention libraries for efficient model training
