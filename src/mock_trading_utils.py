import pandas as pd
import data_paths

def load_data(asset, data_path=None):
    if data_path is None:
        data_path = data_paths.data_paths[asset]
    else:
        return pd.read_csv(data_path)

    if asset == 'topix500':
        df = pd.read_csv(data_path)
        mask = df.timestamp > '2020'
        df = df[mask]
        df = df.pivot(index='timestamp', columns='symbol', values='close')

    elif asset == 'sp500':
        df = pd.read_csv(data_path).set_index('Date')
        df = df[df.index > '2020']
        df = df.dropna(axis=1)

    elif asset == 'forex':
        df = pd.read_csv(data_path).set_index('Date')
        df = df[df.index > '2020']

    elif asset == 'crypto_hourly':
        df = pd.read_csv(data_path)
        df = df.set_index('close_time')
        df = df.ffill() #for prediction, we forward fill
        df = df.dropna(axis=1, how='any')

    elif asset == 'crypto_daily':
        df = pd.read_csv(data_path)
        df = df.set_index('close_time')
        df = df.ffill() #for prediction, we forward fill
        df = df.dropna(axis=1, how='any')
    
    else:
        raise KeyError('Asset must be one of forex/topix500/sp500/crypto_hourly/crypto_daily!')

    return df

def load_data_returns(asset, data_path=None):
    if data_path is None:
        data_path = data_paths.data_paths[asset]
    else:
        return pd.read_csv(data_path)

    if asset == 'forex':
        df = pd.read_csv(data_path).set_index('Date')
        df = df[df.index > '2020']

    elif asset == 'topix500':
        df = pd.read_csv(data_path)
        mask = df.timestamp > '2021'
        df = df[mask]
        df = df.pivot(index='timestamp', columns='symbol', values='close')
        df = df.dropna(axis=1, how='all')
        # some stock splits happened, we want to filter them out
        max_change = (df.diff().shift(-1)/df).abs().max()
        mask = ~(max_change > 0.4)
        df = df.loc[:, mask]

    elif asset == 'sp500':
        df = pd.read_csv(data_path).set_index('Date')
        df = df[df.index > '2020']
        df = df.dropna(axis=1)

    elif asset == 'crypto_daily':
        df = pd.read_csv(data_path)
        df = df.set_index('close_time')
        df = df[pd.to_datetime(df.index).hour == 23]
    return df