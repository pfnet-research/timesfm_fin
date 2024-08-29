# Fine-tuning TimesFM on financial data

## Introduction
[TimesFM](https://github.com/google-research/timesfm)  is a time series foundation model released by Google in 2024. This repo contains code following this [work](https://tech.preferred.jp/en/blog/timesfm/) , fine-tuning TimesFM on financial data, aligning towards the task of price prediction.

## Installation
The `timesfm` package can only be installed in *Python 3.10* due to package conflicts. Ensure that you have the correct Python version installed, which in conda can be done with 

```bash
conda create -n myenv python=3.10
conda activate myenv
```

and then installing the package:

```bash
pip install timesfm
```

To run the AR1 model in `mock_trading.ipynb`, you will also need the `statsmodels` package. 

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
| Linear warmup epochs           | 5 |
| Total epochs                   | 100 |
| Peak learning rate             | 1e-4                              |
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
python src/mock_trading.py --workdir=/path/to/workdir --data_path=/path/to/dataset
```
where `workdir` is where the `positions.csv` file, representing the buy/sell orders of each day, will be stored. `data_path` is the path to your test dataset location. `mock_trading_utils.py` contains some data loading functions but is mainly for PFN internal usage. Please adapt this to your own data cleansing needs. 

## Key benchmarks
For reference, we provide some key performance benchmarks attained by our experimental runs.
We are able to achieve around a 30% of overall train/eval loss reduction. On our test set, we achieve the following performance on S&P500. 

| Horizon | Ann Sharpe | Max Drawdown | Ann Returns | Ann Volatility | Neutral Cost (%) |
|---------|------------|--------------|-------------|----------------|------------------|
| 2       | 0.516     | -0.0015      | 0.0125      | 0.0242         | 0.0025           |
| 4       | -0.482     | -0.0283      | -0.0094      | 0.0194         | -0.0055           |
| 8       | 0.227     | -0.0168      | 0.0049      | 0.0215        | 0.0067           |
| 16      | 0.003     | -0.0189      | 0.0001      | 0.0242         | 0.0002           |
| 32      | 0.420     | -0.0155      | 0.0143      | 0.0339         | 0.0804           |
| 64      | 1.285     | -0.0022      | 0.0333      | 0.0260         | 0.3472           |
| 128     | 1.679    | -0.0009      | 0.0361      | 0.0215         | 0.6005           |

The following is a sharpe ratio comparison between our model and traditional benchmarks.
|                | Ours | Original TimesFM | Random | AR1  |
|----------------|------|------------------|--------|------|
| S&P500         | 1.68 | 0.42             | 0.03   | 1.58 |
| TOPIX500       | 1.06 | -1.75            | 0.11   | -0.82|
| Currencies     | 0.25 | -0.04            | -0.03  | 0.88 |
| Crypto Daily   | 0.26 | -0.03            | 0.01   | 0.17 |

## Weights
Pretrained weight is available: https://huggingface.co/pfnet/timesfm-1.0-200m-fin

