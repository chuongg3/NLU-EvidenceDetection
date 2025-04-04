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

This is a binary classification model that was trained to detect whether two pieces of text (claim and evidence) are related to each other (evidence detection).
This is a deep-learning without transformer architectures approach.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a Siamese model and a multi-headed co-attention mechanism that was trained on 21K pairs of texts and validated on 6K pairs of texts. 

The textual inputs are first pre-processed by removing any instances of '[ref]', '[ref' and 'ref]'. Then every piece of text is encoded by Setence BERT into a 384 dimensional vector each, and passed onto a siamese encoder which is shared by both the claim and evidence embeddings to produce an embedding that is more relevant to the evidence detection task.

Then multi-headed co-attention mechanism models interactions between claims and evidence. This allows the claims to attend to the evidence, and vice versa, each element of one embedding can prioritise the information from different elements of the other embedding to create a more robust understanding before being passed onto a few dense layers before the final predictions using a sigmoid activation function.

The output from the model is a real number between 0-1 inclusive, so the threshold is then tuned on the validation/dev dataset to produce a binary output.

- **Developed by:** Tuan Chuong Goh and Dhruv Sharma
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Siamese Model with Multi-Headed Co-Attention
<!-- - **Finetuned from model [optional]:** [More Information Needed] -->

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/jiasenlu/HieCoAttenVQA
- **Paper or documentation:** https://proceedings.neurips.cc/paper_files/paper/2016/file/9dcb88e0137649590b755372b040afad-Paper.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

30K pairs of texts drawn from emails, news articles and blog posts.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

      
      - num_epochs: 23
      - batch_size: 32
      - learning_rate: 0.007402024937971863
      - dropout0: 0.11597189185804023
      - dropout1: 0.1649105768518277
      - dropout2: 0.24244195005155006
      - units: [512, 64]
      - attention_dropout: 0.10362259259272402
      - attention_dim: 256
      - num_heads: 4
      - ff_dim: 128
      

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 3 minutes
      - duration per training epoch: 30 seconds
      - model size: 5MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A development set was provided with 6K pairs of texts.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy: 0.8447519406
      - Macro Precision: 0.8096910264
      - Macro Recall: 0.7940469822
      - Macro F1-Score: 0.8011873505
      - Weighted Precision: 0.8416590412
      - Weighted Recall: 0.8447519406
      - Weighted F1-Score: 0.8427417504
    

### Results

The model obtained a Macro F1-score of 80% and an accuracy of 84.5%.

## Technical Specifications

### Hardware


      - RAM: at least 16GB
      - Storage: at least 2GB,
      - GPU: RTX 3050Ti

### Software


      - numpy 1.26.4
      - optuna 4.1.0
      - pandas 2.2.3
      - regex 2024.9.11
      - tensorflow 2.17.0
      - scikit-learn 1.5.2
      - sentence_transformers 3.4.1
      

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Training sample was unbalanced, with 72% of the data being negative samples.
Class weightings was used to adjust for this, but model may still be biased towards the majority class.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by bayesian optimisation, and early stopping was applied with a patience of 10 epochs on the validation accuracy.
