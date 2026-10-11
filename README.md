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

The notebooks load the dataset directly from Hugging Face with `load_dataset("amarfurt/art-nlp-hs26", split=...)`, so no manual download is needed.

## How to run it

I used Python 3.12. Create a virtual environment and install the packages:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then open the notebooks with the `.venv` kernel and run them in order:

1. `01_data_exploration.ipynb` – a first look at the data
2. `02_preprocessing.ipynb` – tokenization and choice of word embeddings
3. `03_model1_embeddings.ipynb` – model 1
4. `04_model2_rnn.ipynb` – model 2
5. `05_evaluation.ipynb` – final results on the test set

The GloVe embeddings (about 400 MB) are downloaded automatically the first time. The comparison cell in `02_preprocessing.ipynb` also downloads fastText and word2vec (about 2.7 GB); you can skip it, the results are saved in the notebook.

## References

- Bhagavatula et al. (2020). *Abductive Commonsense Reasoning*. ICLR 2020. https://arxiv.org/abs/1908.05739
- Pennington et al. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP 2014. https://nlp.stanford.edu/projects/glove/

## Use of AI

Claude (Anthropic) and ChatGPT (OpenAI) were used for brainstorming and coding assistance.