# Fine-tuning TimesFM on financial data

## Introduction
[TimesFM](https://github.com/google-research/timesfm)  is a time series foundation model released by Google in 2024. This repo contains code following this [work](https://tech.preferred.jp/en/) , fine-tuning TimesFM on financial data, aligning towards the task of price prediction.

## Installation
The `timesfm` package can only be installed in Python 3.10 due to package conflicts. Ensure that you have the correct Python version installed, and then run the following command:

```bash
pip install timesfm
```

## Data
The fine-tuning dataset is proprietary and not publicly available. However, you can download the necessary data using the following APIs:

- [Binance API](https://github.com/binance/binance-public-data/tree/master/python)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

## Fine-Tuning on Financial Data
To run the code in this repository, use the following command:

```bash
python src/main.py --workdir=/path/to/workdir --config=configs/fine_tuning.py --dataset_path=/path/to/dataset
```

Replace `/path/to/workdir` and `/path/to/dataset` with your local paths.
Logs, tensorboard data and checkpoints will be stored in `workdir`.
`src/fine-tuning.py` contains the necessary configurations for fine-tuning. A brief summary of the hyperparameter settings is found here:

| Hyperparameter/Architecture    | Setting                           |
|--------------------------------|-----------------------------------|
| Optimizer                      | SGD                               |
| Linear warmup epochs           | 25 |
| Total epochs                   | 100 |
| Peak learning rate             | 5e-4                              |
| Momentum                       | 0.9                               |
| Gradient clip (max norm)       | 1.0                               |
| Batch size                     | 1024                              |
| Max context length             | 512                               |
| Min context length             | 128                               |
| Output length                  | 128                               |
| Layers                         | 20                                |
| Hidden dimensions              | 1280                              |

## Mock trading
We provide our mock trading script and notebook used in calculating several evaluation metrics. To run the mock trading script, use the following command 

```bash
python src/mock_trading.py --workdir=/path/to/workdir
```
where `workdir` is where the `positions.csv` file, representing the buy/sell orders of each day, will be stored. 

## Key benchmarks
For reference, we provide some key performance benchmarks attained by our experimental runs.
We are able to achieve around a 30% of overall train/eval loss reduction. On our test set, we achieve the following performance on S&P500. 

| Horizon | Ann Sharpe | Max Drawdown | Ann Returns | Ann Volatility | Neutral Cost (%) |
|---------|------------|--------------|-------------|----------------|------------------|
| 2       | 0.8581     | -0.0099      | 0.0196      | 0.0229         | 0.0039           |
| 4       | 0.3779     | -0.0195      | 0.0097      | 0.0256         | 0.0057           |
| 8       | 0.5604     | -0.0206      | 0.0143      | 0.0256         | 0.0196           |
| 16      | 0.6184     | -0.0150      | 0.0185      | 0.0298         | 0.0527           |
| 32      | 0.8794     | -0.0086      | 0.0263      | 0.0299         | 0.1485           |
| 64      | 1.4738     | -0.0022      | 0.0371      | 0.0252         | 0.3858           |
| 128     | 1.5055     | -0.0010      | 0.0354      | 0.0235         | 0.5892           |

The following is a sharpe ratio comparison between our model and traditional benchmarks.
|                | Ours | Original TimesFM | Random | AR1  |
|----------------|------|------------------|--------|------|
| S&P500         | 1.51 | 0.42             | 0.03   | 0.82 |
| TOPIX500       | 3.42 | -1.62            | 0.11   | -1.15|
| Currencies     | 0.51 | -0.04            | -0.03  | 0.73 |
| Crypto Daily   | 0.04 | -0.58            | 0.01   | 0.70 |
