# TimesFM fine-tuned on financial data

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
Logs and tensorboard data will be stored in workdir. 

## Key benchmarks
