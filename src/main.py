import os
from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import timesfm

import train
import evaluation

import warnings
warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_string('dataset_path', None, 'Path to training/test dataset')
flags.DEFINE_bool('do_eval', False, 'Evaluation mode.')
flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint.')

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)

def main(argv):
  config = FLAGS.config
  workdir = FLAGS.workdir
  do_eval = FLAGS.do_eval
  checkpoint_path = FLAGS.checkpoint_path
  config.dataset_path = FLAGS.dataset_path

  tfm = timesfm.TimesFm(
    context_len=512,
    horizon_len=128, # TODO: why does setting horizon_len to 512 not work
    input_patch_len=32,
    output_patch_len=128, # this is set to the same as horizon length during training
    num_layers=20,
    model_dims=1280,
    backend='gpu'
  )

  if checkpoint_path is not None:
    tfm.load_from_checkpoint(checkpoint_path)
  else:
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
  
  if do_eval:
    evaluation.restore_and_evaluate(tfm, config, workdir)
  else:
    train.train_and_evaluate(tfm, config, workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir', 'dataset_path'])
  app.run(main)