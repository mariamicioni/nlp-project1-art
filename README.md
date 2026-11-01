# NLP HS26 – Project 1: Word Embeddings and RNNs

This project focuses on abductive commonsense reasoning using the ART dataset. Given the beginning and end of a short story, the model selects the more plausible of two possible explanations.

**In short:** I compare averaging and a GRU using the same fixed-position arrangement, selecting each model by validation loss. Additional experiments support retaining the preprocessing and show that per-choice scoring improves model 1 on validation. Final test evaluation will assess both selected models and separately check whether the per-choice improvement generalizes.

**GPU run (1.11):** the code of `03`, `04` and `04b` now uses a GPU if there is one, patience 10 for every run, and the largest batch size that fits the GPU, up to 2048. The results in this README and in the notebooks are still those of the CPU run (commit `e5aaca6`) until these notebooks are run again. W&B project of the GPU run: `nlp-project1-art-gpu`.

## Models

Two models are compared:

1. **Model 1:** Frozen pretrained word embeddings + 2-layer classifier (ReLU). Only the classifier is trained.
2. **Model 2:** Same frozen pretrained word embeddings + 2-layer LSTM/GRU + 2-layer classifier (ReLU). Both the RNN and classifier are trained.

The pretrained embeddings are GloVe with 300 dimensions (`glove-wiki-gigaword-300`, loaded with gensim).

## Data

This project uses the HSLU course version of the ART dataset, available on [Hugging Face](https://huggingface.co/datasets/amarfurt/art-nlp-hs26).

The dataset contains:

- **Training:** 16,801 examples
- **Validation:** 1,000 examples
- **Test:** 1,532 examples

The training set is used to train the models, the validation set for hyperparameter tuning and model selection, and the test set for final evaluation.

The course made its own splits from the original dataset [`allenai/art`](https://huggingface.co/datasets/allenai/art) (course slides "Course Project HS26", slide 8), and they differ in difficulty. I compared the texts of the examples with `allenai/art`: all 16,801 training and all 1,000 validation examples occur in its train split, and all 1,532 test examples in its validation split. The paper says about the original dataset: "The train set includes all plausible and implausible hypotheses collected via crowdsourcing, while the dev and test sets include the hypotheses selected through the Adversarial Filtering algorithm" (Bhagavatula et al., 2020, Table 6). So the course training and validation data are unfiltered and the test data is adversarially filtered, and the validation accuracy is expected to be higher than the test accuracy.

The notebooks load the dataset directly from Hugging Face with `load_dataset("amarfurt/art-nlp-hs26", split=...)`, so no manual download is needed.

## How to run it

I used Python 3.12. Create a virtual environment and install the packages:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The training notebooks (`03`, `04`, `04b` and `04c`) log every run to Weights & Biases (W&B). Log in once in the terminal with the API key of your own (free) W&B account; the key is stored locally, and the notebooks then log in with `wandb.login()` without asking:

```bash
wandb login
```

Then open the notebooks with the `.venv` kernel and run them in order:

1. `01_data_exploration.ipynb` – a first look at the data
2. `02_preprocessing.ipynb` – preprocessing decisions, choice of word embeddings and the input of both models
3. `03_model1_embeddings.ipynb` – model 1: training, experiments, results
4. `04_model2_rnn.ipynb` – model 2: training, experiments, results
5. `04b_preprocessing_experiments.ipynb` – additional check, not required by the project: the preprocessing decisions and the input format tested by training, one option at a time, with the chosen settings of `03` and `04`; the test set is not loaded
6. `04c_patience_check.ipynb` – additional check, not required by the project: the chosen model 2 with patience 10 instead of 3; no other notebook depends on it
7. `05_evaluation.ipynb` – the chosen model of each architecture on the test set, the hypothesis-only runs as diagnostics, and per-choice scoring of model 1 as a separate comparison

The GloVe embeddings (about 400 MB) are downloaded automatically the first time. The comparison cell in `02_preprocessing.ipynb` also downloads fastText and word2vec (about 2.7 GB); you can skip it, the results are saved in the notebook.

`02_preprocessing.ipynb` saves the input features of model 1 to `features/model1.npz` (about 93 MB), which `03_model1_embeddings.ipynb` loads, and the input of model 2 (padded token indices and the GloVe embedding matrix) to `features/model2.npz` (about 40 MB), which `04_model2_rnn.ipynb` loads. The files are not in the repository because they can be recreated, so run `02_preprocessing.ipynb` before the training notebooks.

Every training run saves the weights of its best epoch to `checkpoints/` (not in the repository either); `05_evaluation.ipynb` loads the checkpoints of the chosen runs, of the hypothesis-only runs and of the per-choice run, so run `03`, `04` and `04b` before it. Every run has a fixed seed: 42, or 1 and 2 in the sections "Other seeds". A run gives the same results when it is run again on the same machine. All runs were trained on my laptop CPU (Intel Core i7-1360P, no GPU): 38 in `03_model1_embeddings.ipynb`, 28 in `04_model2_rnn.ipynb`, 34 in `04b_preprocessing_experiments.ipynb` and 1 in `04c_patience_check.ipynb`.

## Results

All choices were made on the validation set; the test set is evaluated once, in `05_evaluation.ipynb`. The primary comparison uses the fixed-position input format. For each architecture, the run and checkpoint are selected by validation loss from the grid and sweep experiments in notebooks 03 and 04. The walk-through, diagnostic and seed runs are not candidates. The variants in notebook 04b are additional experiments and do not replace these selected models. Per-choice scoring improves model 1 on validation and is also evaluated on the test set as a separate comparison.

| Model | Chosen configuration | Best epoch | Validation loss | Validation accuracy |
|---|---|---|---|---|
| Model 1: averaged GloVe vectors + 2-layer classifier | 100 hidden units, learning rate 0.000316, batch size 32 | 13 | 0.6445 | 63.20% |
| Model 2: GloVe + 2-layer RNN + 2-layer classifier | GRU, RNN hidden size 128, learning rate 0.001, dropout 0.5 | 4 | 0.5872 | 67.60% |

Separate comparison, not a selected model: model 1 with per-choice scoring, with the same settings, best epoch 32, validation loss 0.5829, validation accuracy 69.10%.

**Test evaluation pending.**

Experiments: model selection used 32 candidate runs for model 1 and 26 for model 2, in grids and W&B sweeps. Notebook `04b` reports 34 additional preprocessing and input-format runs (17 per model): no preprocessing variant meets the decision rule stated there, so the preprocessing of `02` stays; one input format variant does (`per-choice` for model 1). The chosen configurations are also trained with seeds 1 and 2, and these runs are only reported. Notebook `04c` repeats the chosen model 2 with patience 10: its best epoch stays epoch 4. The sweep winner of model 1 is not shown to be better than the best run of the first grid (exact McNemar test, p = 0.8546). Detailed configurations and results are in the notebooks, sections "Results" and "Conclusion", and in W&B. The expected test accuracies, written down before the experiments, are in `EXPECTATIONS.md` (model 1: 52%, model 2: 53%).

## References

### Course materials

Slides and exercise notebooks of the course NLP HS26 (A. Marfurt, Lucerne University of Applied Sciences and Arts), available to course participants on Ilias. The notebooks cite them in short form, e.g. "course exercise "PyTorch" (Week 1)".

Lecture slides:

- Marfurt, A. (2026, September 15). *Course Project HS26* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Project 1: Word Embeddings/Recurrent Neural Networks* [Project description]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 15). *Preliminaries* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 17). *NLP Pipeline* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 22). *Vector Space Model* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 24). *Word Embeddings* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, September 29). *Recurrent Neural Networks* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026, October 6). *Advanced RNNs* [Lecture slides]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Grading criteria HS26* [Course document]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Project Discussions* [Lecture slides, weeks 1–7]. NLP HS26, Lucerne University of Applied Sciences and Arts.

Exercise notebooks:

- Marfurt, A. (2026). *NLP Pipeline* [Exercise solution, `nlp_pipeline_solution.ipynb`, Week 1]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *PyTorch* [Exercise solution, `pytorch_solution.ipynb`, Week 1]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Topic Modeling* [Exercise solution, `topic_modeling_solution.ipynb`, Week 2]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Word Embeddings* [Exercise solution, `word2vec_solution.ipynb`, Week 2]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Experiment Tracking* [Exercise solution, `experiment_tracking_solution.ipynb`, Project Discussions]. NLP HS26, Lucerne University of Applied Sciences and Arts.
- Marfurt, A. (2026). *Sequence-to-sequence RNN* [Exercise solution, `seq2seq_rnn_solution.ipynb`, Week 4]. NLP HS26, Lucerne University of Applied Sciences and Arts.

### Data and papers

- Marfurt, A. (2026). *art-nlp-hs26* [Dataset]. Hugging Face. https://huggingface.co/datasets/amarfurt/art-nlp-hs26
- Bhagavatula, C., Le Bras, R., Malaviya, C., Sakaguchi, K., Holtzman, A., Rashkin, H., Downey, D., Yih, W., & Choi, Y. (2020). Abductive Commonsense Reasoning. *International Conference on Learning Representations (ICLR 2020)*. https://arxiv.org/abs/1908.05739
- Mostafazadeh, N., Chambers, N., He, X., Parikh, D., Batra, D., Vanderwende, L., Kohli, P., & Allen, J. (2016). A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories. *Proceedings of NAACL-HLT 2016*, 839–849.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *Proceedings of EMNLP 2014*, 1532–1543. https://nlp.stanford.edu/projects/glove/
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *Advances in Neural Information Processing Systems 26 (NeurIPS 2013)*.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *Transactions of the Association for Computational Linguistics, 5*, 135–146.
- Lu, L., Shin, Y., Su, Y., & Karniadakis, G. E. (2020). Dying ReLU and Initialization: Theory and Numerical Examples. *Communications in Computational Physics*. https://doi.org/10.4208/cicp.OA-2020-0165
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html
- Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent Neural Network Regularization. *arXiv preprint arXiv:1409.2329*. https://arxiv.org/abs/1409.2329
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv preprint arXiv:1607.06450*. https://arxiv.org/abs/1607.06450
- Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S. R., & Smith, N. A. (2018). Annotation Artifacts in Natural Language Inference Data. *Proceedings of NAACL-HLT 2018*. https://arxiv.org/abs/1803.02324
- Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., & Van Durme, B. (2018). Hypothesis Only Baselines in Natural Language Inference. *Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics (\*SEM 2018)*, 180–191. https://arxiv.org/abs/1805.01042
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1711.05101
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Hägele, A., Bakouch, E., Kosson, A., Ben Allal, L., Von Werra, L., & Jaggi, M. (2024). Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. *arXiv preprint arXiv:2405.18392*. https://arxiv.org/abs/2405.18392 (cited on slide 33 of the course slides "Project Discussions")

### Software

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. (NLTK) https://www.nltk.org/
- Řehůřek, R., & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, 45–50. (gensim) https://radimrehurek.com/gensim/
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. https://pytorch.org/
- Lhoest, Q., Villanova del Moral, A., Jernite, Y., et al. (2021). Datasets: A Community Library for Natural Language Processing. *Proceedings of EMNLP 2021: System Demonstrations*, 175–184. (Hugging Face Datasets)
- McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. (pandas)
- Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering, 9*(3), 90–95.
- Biewald, L. (2020). *Experiment Tracking with Weights and Biases*. Software available from https://wandb.ai/
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods, 17*, 261–272. https://doi.org/10.1038/s41592-019-0686-2

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
- PyTorch documentation: `torch.nn.GRU` (also the variable `weight_ih_l[k]`), https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- PyTorch tutorial: "Datasets & DataLoaders" (`next(iter(...))`), https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- Python documentation: built-in function `isinstance`, https://docs.python.org/3/library/functions.html#isinstance
- PyTorch documentation: `torch.nn.LSTM` (`h_n`, the final hidden state), https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch documentation: `torch.nn.utils.rnn.pack_padded_sequence`, https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
- NumPy documentation: `numpy.sum` (`axis`), https://numpy.org/doc/stable/reference/generated/numpy.sum.html
- PyTorch documentation: `torch.sum` (`dim`), https://docs.pytorch.org/docs/stable/generated/torch.sum.html
- PyTorch tutorial "Neural Networks": https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- PyTorch tutorial "Optimizing Model Parameters": https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- PyTorch documentation: `torch.nn.CrossEntropyLoss`, https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch documentation: "Reproducibility" (`torch.manual_seed`), https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch documentation: `torch.utils.data` (`DataLoader`, `TensorDataset`), https://docs.pytorch.org/docs/stable/data.html
- PyTorch documentation: `torch.mean`, https://docs.pytorch.org/docs/stable/generated/torch.mean.html
- scikit-learn glossary (naming convention `X` and `y`): https://scikit-learn.org/stable/glossary.html
- Weights & Biases quickstart (`wandb.init` with `project` and `config`): https://docs.wandb.ai/models/quickstart
- Weights & Biases documentation: `wandb.init()` reference (`project`, `name`, `config`), https://docs.wandb.ai/models/ref/python/functions/init
- Weights & Biases documentation: "Log objects and media" (`run.log`, `epoch` as custom x-axis), https://docs.wandb.ai/models/track/log
- PyTorch tutorial "Save and Load the Model": https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Python documentation: built-in function `float` (`float("inf")`), https://docs.python.org/3/library/functions.html#float
- Python documentation: built-in functions `all()`, `any()`, `ord()`, `zip()`, `enumerate()` and `round()`, https://docs.python.org/3/library/functions.html
- Python documentation: conditional expressions, https://docs.python.org/3/reference/expressions.html#conditional-expressions
- Python documentation: `string.punctuation`, https://docs.python.org/3/library/string.html#string.punctuation
- Python documentation: `re` (regular expressions, `re.search`), https://docs.python.org/3/library/re.html
- NLTK documentation: "Installing NLTK Data" (`nltk.download`), https://www.nltk.org/data.html, and `nltk.probability.FreqDist`, https://www.nltk.org/api/nltk.probability.FreqDist.html
- Weights & Biases documentation: "Log summary metrics" (`run.summary`), https://docs.wandb.ai/models/track/log/log-summary
- Python documentation: "Sorting Techniques" (`sorted()`), https://docs.python.org/3/howto/sorting.html
- Karpathy, A. (2019, April 25). *A Recipe for Training Neural Networks* [Blog post] ("overfit one batch"). https://karpathy.github.io/2019/04/25/recipe/
- PyTorch documentation: `torch.equal`, https://docs.pytorch.org/docs/stable/generated/torch.equal.html
- PyTorch documentation: `torch.nn.Dropout`, https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- PyTorch documentation: `torch.nn.LayerNorm`, https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- PyTorch documentation: `torch.numel`, https://docs.pytorch.org/docs/stable/generated/torch.numel.html
- PyTorch documentation: `torch.nn.Sequential`, https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html
- Python documentation: tutorial "Classes" (inheritance), https://docs.python.org/3/tutorial/classes.html#inheritance
- PyTorch documentation: `torch.optim.AdamW` (algorithm), https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- Python documentation: `time` module (`time.time()`), https://docs.python.org/3/library/time.html
- Python documentation: set types (`^` and `<=`), https://docs.python.org/3/builtins/stdtypes.html#set-types-set-frozenset
- Python documentation: `str.split`, `str.lower` and `list.extend`, https://docs.python.org/3/library/stdtypes.html
- NumPy documentation: `ndarray.min`, https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html, `ndarray.max`, https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html, and `ndarray.std`, https://numpy.org/doc/stable/reference/generated/numpy.ndarray.std.html
- pandas documentation: `pandas.set_option`, https://pandas.pydata.org/docs/reference/api/pandas.set_option.html
- Matplotlib documentation: `pyplot.subplots`, https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html, and `Axes.hist`, https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
- Python tutorial: "Comparing Sequences and Other Types" (`==` on lists), https://docs.python.org/3/tutorial/datastructures.html#comparing-sequences-and-other-types
- Python documentation: the `assert` statement, https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement
- PyTorch documentation: `torch.bitwise_and`, https://docs.pytorch.org/docs/stable/generated/torch.bitwise_and.html, and `torch.bitwise_not`, https://docs.pytorch.org/docs/stable/generated/torch.bitwise_not.html
- PyTorch documentation: `torch.nonzero`, https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html
- SciPy documentation: `scipy.stats.binomtest`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
- Wikipedia: "McNemar's test" (exact binomial test), https://en.wikipedia.org/wiki/McNemar%27s_test
- Wikipedia: "Holm–Bonferroni method", https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
- Weights & Biases documentation (now at docs.coreweave.com): sweep configuration, https://docs.coreweave.com/models/sweeps/define-sweep-configuration; its options, https://docs.coreweave.com/models/sweeps/sweep-config-keys; sweeps walkthrough, https://docs.coreweave.com/models/sweeps/walkthrough; run config, https://docs.coreweave.com/models/track/config; `Run`, https://docs.coreweave.com/models/ref/python/experiments/run
- NLTK documentation: `nltk.tag` (`pos_tag`), https://www.nltk.org/api/nltk.tag.html; `PerceptronTagger`, https://www.nltk.org/api/nltk.tag.perceptron.html; `WordNetLemmatizer.lemmatize`, https://www.nltk.org/api/nltk.stem.wordnet.html; `upenn_tagset`, https://www.nltk.org/api/nltk.help.html
- PyTorch documentation: `torch.clamp`, https://docs.pytorch.org/docs/stable/generated/torch.clamp.html
- NumPy documentation: `numpy.transpose`, https://numpy.org/doc/stable/reference/generated/numpy.transpose.html, and `numpy.array_equal`, https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
- Python documentation: `dict.get`, https://docs.python.org/3/library/stdtypes.html#dict.get
- scikit-learn documentation: "Permutation feature importance", https://scikit-learn.org/stable/modules/permutation_importance.html
- Weights & Biases documentation (now at docs.coreweave.com): "Environment variables" (`WANDB_SILENT`), https://docs.coreweave.com/models/track/environment-variables
- Python documentation: `os.environ`, https://docs.python.org/3/library/os.html#os.environ
- PyTorch documentation: `torch.flip`, https://docs.pytorch.org/docs/stable/generated/torch.flip.html, `torch.randn`, https://docs.pytorch.org/docs/stable/generated/torch.randn.html, and `torch.max`, https://docs.pytorch.org/docs/stable/generated/torch.max.html
- NumPy documentation: `numpy.random.default_rng` and `Generator.normal`, https://numpy.org/doc/stable/reference/random/generator.html, and "Indexing on ndarrays", https://numpy.org/doc/stable/user/basics.indexing.html
- PyTorch documentation: `torch.randperm`, https://docs.pytorch.org/docs/stable/generated/torch.randperm.html, and `Tensor.clone`, https://docs.pytorch.org/docs/stable/generated/torch.Tensor.clone.html

## Use of AI

I used the following AI tools. I reviewed the code and the results, and I can explain them.

- **Claude (Anthropic):** help with coding and documentation; finding and checking sources for the citations.
- **ChatGPT (OpenAI):** reviews of the project, general questions, and style checks; its corrections were checked before they were adopted.
- **Gemini (Google):** a review of the code and the experiments; its suggestions were checked, and only some of them were adopted.