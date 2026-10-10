# NLP HS26 – Project 1: Word Embeddings and RNNs

This project focuses on abductive commonsense reasoning using the ART dataset. Given the beginning and end of a short story, the model selects the more plausible of two possible explanations.

## Models

Two models are compared:

1. **Model 1:** Frozen pretrained word embeddings + 2-layer classifier (ReLU). Only the classifier is trained.
2. **Model 2:** Same frozen pretrained word embeddings + 2-layer LSTM/GRU + 2-layer classifier (ReLU). Both the RNN and classifier are trained.

## Data

This project uses the HSLU course version of the ART dataset, available on [Hugging Face](https://huggingface.co/datasets/amarfurt/art-nlp-hs26).

The dataset contains:

- **Training:** 16,801 examples
- **Validation:** 1,000 examples
- **Test:** 1,532 examples

The training set is used to train the models, the validation set for hyperparameter tuning and model selection, and the test set for final evaluation.

To download the dataset into the `data/` folder, run the following command from the project root:

```bash
git clone https://huggingface.co/datasets/amarfurt/art-nlp-hs26 data
```

## References

- Bhagavatula et al. (2020). *Abductive Commonsense Reasoning*. ICLR 2020. https://arxiv.org/abs/1908.05739

## Use of AI

Claude (Anthropic) and ChatGPT (OpenAI) were used for brainstorming and coding assistance.