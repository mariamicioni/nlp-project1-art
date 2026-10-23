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
2. `02_preprocessing.ipynb` – tokenization, choice of word embeddings and the input features of model 1
3. `03_model1_embeddings.ipynb` – model 1
4. `04_model2_rnn.ipynb` – model 2
5. `05_evaluation.ipynb` – final results on the test set

The GloVe embeddings (about 400 MB) are downloaded automatically the first time. The comparison cell in `02_preprocessing.ipynb` also downloads fastText and word2vec (about 2.7 GB); you can skip it, the results are saved in the notebook.

`02_preprocessing.ipynb` saves the input features of model 1 to `features/model1.npz` (about 93 MB), which `03_model1_embeddings.ipynb` loads. The file is not in the repository because it can be recreated, so run `02_preprocessing.ipynb` before `03_model1_embeddings.ipynb`.

## References

### Course materials

Slides and exercise notebooks of the course NLP HS26 (A. Marfurt, Lucerne University of Applied Sciences and Arts), available to course participants on Ilias. The notebooks cite them in short form, e.g. "course exercise "PyTorch" (Week 1)".

Lecture slides:

- Marfurt, A. (2026, September 15). *Course Project HS26* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 15). *Preliminaries* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 22). *Vector Space Model* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 24). *Word Embeddings* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 29). *Recurrent Neural Networks* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, October 6). *Advanced RNNs* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.

Exercise notebooks:

- Marfurt, A. (2026). *NLP Pipeline* [Exercise solution, `nlp_pipeline_solution.ipynb`, Week 1]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *PyTorch* [Exercise solution, `pytorch_solution.ipynb`, Week 1]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Topic Modeling* [Exercise solution, `topic_modeling_solution.ipynb`, Week 2]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Word Embeddings* [Exercise solution, `word2vec_solution.ipynb`, Week 2]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Experiment Tracking* [Exercise solution, `experiment_tracking_solution.ipynb`, Project Discussions]. NLP HS26, Lucerne University of Applied Sciences and Arts.

### Data and papers

- Marfurt, A. (2026). *art-nlp-hs26* [Dataset]. Hugging Face. https://huggingface.co/datasets/amarfurt/art-nlp-hs26
- Bhagavatula, C., Le Bras, R., Malaviya, C., Sakaguchi, K., Holtzman, A., Rashkin, H., Downey, D., Yih, W., & Choi, Y. (2020). Abductive Commonsense Reasoning. *International Conference on Learning Representations (ICLR 2020)*. https://arxiv.org/abs/1908.05739
- Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., Kohli, P., & Allen, J. (2016). A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories. *Proceedings of NAACL-HLT 2016*, 839–849.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *Proceedings of EMNLP 2014*, 1532–1543. https://nlp.stanford.edu/projects/glove/
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *Advances in Neural Information Processing Systems 26 (NeurIPS 2013)*.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *Transactions of the Association for Computational Linguistics, 5*, 135–146.
- Lu, L., Shin, Y., Su, Y., & Karniadakis, G. E. (2020). Dying ReLU and Initialization: Theory and Numerical Examples. *Communications in Computational Physics*. https://doi.org/10.4208/cicp.OA-2020-0165

### Software

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. (NLTK) https://www.nltk.org/
- Řehůřek, R., & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, 45–50. (gensim) https://radimrehurek.com/gensim/
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. https://pytorch.org/
- Lhoest, Q., Villanova del Moral, A., Jernite, Y., et al. (2021). Datasets: A Community Library for Natural Language Processing. *Proceedings of EMNLP 2021: System Demonstrations*, 175–184. (Hugging Face Datasets)
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. (pandas)
- Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering, 9*(3), 90–95.
- Biewald, L. (2020). *Experiment Tracking with Weights and Biases*. Software available from https://wandb.ai/

### Documentation and tutorials (accessed October 2026)

- NumPy documentation: `numpy.savez`, https://numpy.org/doc/stable/reference/generated/numpy.savez.html, and `numpy.ndarray.astype`, https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
- Python documentation: `os` module, https://docs.python.org/3/library/os.html
- NumPy documentation: `numpy.percentile`, https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
- Python documentation: "Common Sequence Operations" (slicing, `*` and `+` on lists), https://docs.python.org/3/library/stdtypes.html#common-sequence-operations
- Python tutorial: "Dictionaries", https://docs.python.org/3/tutorial/datastructures.html#dictionaries
- Python tutorial: "List Comprehensions" (with an `if` condition), https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
- NumPy documentation: `numpy.zeros` (`shape`, `dtype`), https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
- Python tutorial: "Looping Techniques" (`dict.items()`), https://docs.python.org/3/tutorial/datastructures.html#looping-techniques
- PyTorch documentation: `torch.nn.Embedding` (`from_pretrained`, `freeze`, `padding_idx`), https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- PyTorch documentation: `torch.nn.LSTM` (`h_n`, the final hidden state), https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch tutorial "Neural Networks": https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- PyTorch tutorial "Optimizing Model Parameters": https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch documentation: `torch.nn.CrossEntropyLoss`, https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch documentation: "Reproducibility" (`torch.manual_seed`), https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch documentation: `torch.utils.data` (`DataLoader`, `TensorDataset`), https://docs.pytorch.org/docs/stable/data.html
- PyTorch documentation: `torch.mean`, https://docs.pytorch.org/docs/stable/generated/torch.mean.html
- Wikipedia: "Binomial proportion confidence interval" (normal approximation, standard error √(p(1 − p)/n)), https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
- scikit-learn glossary (naming convention `X` and `y`): https://scikit-learn.org/stable/glossary.html
- Weights & Biases quickstart (`wandb.init` with `project` and `config`): https://docs.wandb.ai/models/quickstart
- Weights & Biases documentation: `wandb.init()` reference (`project`, `name`, `config`), https://docs.wandb.ai/models/ref/python/functions/init
- Weights & Biases documentation: "Log objects and media" (`run.log`, `epoch` as custom x-axis), https://docs.wandb.ai/models/track/log
- PyTorch tutorial "Save and Load the Model": https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Python documentation: built-in function `float` (`float("inf")`), https://docs.python.org/3/library/functions.html#float
- Weights & Biases documentation: "Log summary metrics" (`run.summary`), https://docs.wandb.ai/models/track/log/log-summary
- Python documentation: "Sorting Techniques" (`sorted()`), https://docs.python.org/3/howto/sorting.html

## Use of AI

Claude (Anthropic) and ChatGPT (OpenAI) were used for brainstorming and coding assistance.