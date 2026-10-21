# src/model/rnn — Recurrent subnets

The temporal-modeling stage of the encoder. Provides two interchangeable recurrent
subnets; which one is used depends on the `model.name` variant selected in
`config/model_ablation/`.

## Files

| File | Role | Source |
|---|---|---|
| `agc_lstm.py` | `AGCLSTM` — Attention-Enhanced Graph Convolutional LSTM: a graph convolution feeds an LSTM whose hidden state is spatially attention-gated. | AGC-LSTM — Si, Chen, Wang, Wang, Tan, "An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition," CVPR 2019. |
| `lstm_part.py` | `LSTM_part` — bidirectional LSTM wrapper over the per-frame graph features. | LSTM (Hochreiter & Schmidhuber, 1997); in-house wrapper. |
