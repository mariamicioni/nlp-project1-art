# Expectations

*Written on 10.10.2026, before running any experiments.*

Based on Bhagavatula et al. (2020, Table 1), I used the following ART accuracy results as reference points:

| Model | Accuracy |
|---|---|
| Random | 50.4% |
| InferSent | 50.8% |
| ESIM+ELMo | 58.8% |
| BERT | 68.9% |
| Humans | 91.4% |

I expect both models to perform slightly above random but below ESIM+ELMo.

**Model 1: Expected test accuracy = 52%**

This model averages pretrained, frozen word embeddings and passes the representation to a 2-layer classifier with ReLU. Only the classifier is trained. Since averaging ignores word order (Vector Space Model slides, p. 7), it may struggle to distinguish between similar ART hypotheses.

**Model 2: Expected test accuracy = 53%**

This model uses the same frozen embeddings but processes them through a 2-layer LSTM/GRU before applying a 2-layer classifier with ReLU. Both the RNN and classifier are trained. Unlike Model 1, it captures word order and sequential dependencies (Advanced RNNs slides, p. 4). I therefore expect slightly better performance, although the relatively small training set (16,801 examples) may limit improvement.

## Sources

- Bhagavatula et al. (2020). *Abductive Commonsense Reasoning*. https://arxiv.org/abs/1908.05739
- NLP HS26 course slides (A. Marfurt): *Vector Space Model*, *Recurrent Neural Networks*, *Advanced RNNs*.

## Use of AI

I used Claude (Anthropic) and ChatGPT (OpenAI) to help formulate and improve the reasoning. The expected accuracies (52% and 53%) are my own estimates.
