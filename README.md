# NLP HS26 – Project 1: Word Embeddings and RNNs

Abductive reasoning on the ART dataset: given the start and end of a short story, choose the more plausible of two explanations. This branch holds the GPU run (Tesla T4 on Google Colab).

## Models

1. **Model 1:** frozen GloVe vectors (`glove-wiki-gigaword-300`), averaged per text, + 2-layer ReLU classifier.
2. **Model 2:** the same frozen vectors + 2-layer LSTM or GRU + 2-layer ReLU classifier.

## Data

Course version of ART on [Hugging Face](https://huggingface.co/datasets/amarfurt/art-nlp-hs26), loaded by the notebooks: 16,801 training, 1,000 validation and 1,532 test examples. The course made these splits from [`allenai/art`](https://huggingface.co/datasets/allenai/art) (course slides "Course Project HS26", slide 8). I compared the texts: the training and validation examples occur in its train split, which is unfiltered, and the test examples in its validation split, which is adversarially filtered (Bhagavatula et al., 2020, Table 6). So validation accuracy is expected to be higher than test accuracy.

## How to run

On Google Colab with a T4 GPU:

1. Copy this branch to `MyDrive/nlp-project1-art` on Google Drive.
2. Add your W&B API key as the Colab secret `WANDB_API_KEY`.
3. Run the notebooks in this order. `features/` and `checkpoints/` are saved on Drive, not in git.

| Notebook | What it does |
|---|---|
| `01_data_exploration.ipynb` | First look at the data |
| `02_preprocessing.ipynb` | Preprocessing decisions; saves the input of both models |
| `03_model1_embeddings.ipynb` | Model 1: experiments, selection, seeds |
| `04_model2_rnn.ipynb` | Model 2: experiments, selection, seeds |
| `04b_preprocessing_experiments.ipynb` | Optional checks by training, without test data. `ALL_VARIANTS = False`: the 5 runs that `05` needs; `True`: all 48 |
| `05_evaluation.ipynb` | The test set, once |

Locally (Python 3.12): `python -m venv .venv`, activate it, `pip install -r requirements.txt`, `wandb login`. The chosen runs need a GPU: on a CPU, `03` uses batch size 32 and stops at its check. Every run has a fixed seed, and repeated runs on the T4 gave the same results.

## Results

Each model is selected by the lowest validation loss among its candidate runs (31 for model 1, 29 for model 2; fixed-position input, seed 42). The test set is evaluated once, in `05`.

| Model | Chosen configuration | Best epoch | Validation loss | Validation accuracy | Mean of 5 seeds | Test accuracy |
|---|---|---|---|---|---|---|
| Model 1 | 600 hidden units, learning rate 0.000316, batch size 2048 | 40 | 0.6460 | 64.00% | 0.6463 / 63.16% | pending |
| Model 2 | GRU, layer norm in the classifier, RNN hidden size 128, learning rate 0.001, dropout 0.5, batch size 128 | 6 | 0.5867 | 66.70% | 0.5942 / 67.28% | pending |

Reported only, not candidates: other seeds, the next values of model 1, the hypothesis-only runs and the variants of `04b`. Per-choice scoring of model 1 meets the decision rule of `04b` (0.5858, 68.90%, p = 0.0029) and is evaluated in `05` as a separate comparison. Details are in the conclusion of each notebook.

Expected test accuracies, written before the experiments (`EXPECTATIONS.md`): model 1 52%, model 2 53%.

W&B: [saved view of the candidate runs](https://forge.coreweave.com/wandb/maria-micioni-hochschule-luzern/nlp-project1-art-gpu/workspace?nw=8wrb7u99ny1).

## References

Library documentation, tutorials and blog posts (accessed October and November 2026) are cited with their links in the Sources cell of each notebook.

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
- Weights & Biases quickstart (`wandb.init` with `project` and `config`), accessed October 2026. https://docs.wandb.ai/models/quickstart

## Use of AI

I used the following AI tools. I reviewed the code and the results, and I can explain them.

- **Claude (Anthropic):** help with coding and documentation; finding and checking sources for the citations.
- **ChatGPT (OpenAI):** reviews of the project, general questions, and style checks; its corrections were checked before they were adopted.
- **Gemini (Google):** a review of the code and the experiments; its suggestions were checked, and only some of them were adopted.
