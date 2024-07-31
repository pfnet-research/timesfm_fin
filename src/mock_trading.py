from absl import flags
from absl import app
import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import timesfm
import mock_trading_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Directory to checkpoint')
flags.DEFINE_bool('plus_one', True, 'Whether to add 1 to data')
flags.DEFINE_bool('gpu', True, 'Whether to use gpu')
flags.DEFINE_bool('use_log', True, 'Whether to log data')
flags.DEFINE_string('asset', 'sp500', 'One of forex/topix500/sp500/crypto_hourly/crypto_daily')
flags.DEFINE_string('workdir', None, 'Directory to store position csv file')
flags.DEFINE_string('data_path', None, 'Directory to price data to input to model')
flags.DEFINE_integer('horizon', None, 'Prediction horizon length')

context_len = 512
output_len = 128
devices = 8


def main(argv):
    asset = FLAGS.asset
    checkpoint_path =  FLAGS.checkpoint_path
    plus_one = FLAGS.plus_one if checkpoint_path is not None else False
    use_log = FLAGS.use_log if checkpoint_path is not None else False
    workdir = FLAGS.workdir
    data_path = FLAGS.data_path
    if workdir is None:
        workdir = os.path.abspath(os.path.join(os.getcwd(), 'mock_trading'))
    print('Asset:', asset)
    df = mock_trading_utils.load_data(asset=asset, data_path=data_path)

    prediction_dates = df[df.index > '2023'].index

    backend = 'gpu' if FLAGS.gpu else 'cpu'

    tfm = timesfm.TimesFm(
        context_len=512,
        horizon_len=128, 
        input_patch_len=32,
        output_patch_len=128, 
        num_layers=20,
        model_dims=1280,
        backend=backend
    )

    tfm.load_from_checkpoint(checkpoint_path)

    if FLAGS.horizon is not None:
        intervals = [FLAGS.horizon]
    else:
        intervals = [2, 4, 8, 16, 32, 64, 128]

    if checkpoint_path is not None:
        checkpoint_file = os.path.basename(checkpoint_path.rstrip('/'))
        save_folder = os.path.join(workdir, checkpoint_file)
    else:
        save_folder = os.path.join(workdir, 'original')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print('Files saved to:', save_folder)

    for interval in intervals:
        print('Currently running interval:', interval)
        all_positions = []
        for date in prediction_dates:
            df_in = df[df.index < date]
            df_in = df_in.tail(context_len)
            input_ts = df_in.T.to_numpy()
            if plus_one:
                input_ts += jnp.ones_like(input_ts)
            if use_log:
                input_ts = jnp.log(input_ts)
            predictions, quantiiles = tfm.forecast(input_ts, freq=jnp.zeros(input_ts.shape[0]))
            if use_log:
                predictions = jnp.exp(predictions)
            if plus_one:
                predictions -= jnp.ones_like(predictions)
            predictions_cut = predictions[:, :interval]
            start = predictions_cut[:, :1]
            end = predictions_cut[:, -1:]
            positions = jnp.where(end > start, 1, -1) # buy if predict to increase, sell otherwise
            all_positions.append(positions)

        pos = jnp.concatenate(all_positions, axis=1)
        pos = np.asarray(pos).T
        pos = pd.DataFrame(pos, index=prediction_dates, columns=df.columns)

        pos_save_path = os.path.join(save_folder, f'{asset}_{interval}_positions.csv')
        pos.to_csv(pos_save_path)

if __name__ == '__main__':
    app.run(main)