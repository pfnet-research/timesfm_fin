import logging
import time
from typing import Any, Optional, Tuple
import functools
import gc
import datetime
import os

import timesfm
import gc
import numpy as np
import pandas as pd
from timesfm import patched_decoder
from timesfm import data_loader

import jax
from jax import numpy as jnp
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import base_model
from praxis import optimizers
from praxis import schedules
from praxis import base_hyperparams
from praxis import base_layer
from paxml import tasks_lib
from paxml import trainer_lib
from paxml import checkpoints
from paxml import learners
from paxml import partitioning
from paxml import checkpoint_types

from clu import metric_writers
import tensorflow as tf

from utils import mse, get_accuracy, get_returns, get_confusion_matrix
from train import preprocess_csv, prepare_batch_data, eval_step, PatchedDecoderFinetuneFinance, build_learner, reshape_batch, postprocess_metrics

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
instantiate = base_hyperparams.instantiate

_INPUT_TS = "input_ts"
_TARGET_FUTURE = "actual_ts"
_INPUT_PADDING = "input_padding"
_OUTPUT_TS = "output_ts"
_FREQ = "freq"
_OUTPUT_TOKENS = "output_tokens"
_STATS = "stats"

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

np.random.seed(0)

def restore_and_evaluate(
    model: Any, config: py_utils.NestedMap, workdir: str, num_classes=2, plus_one=True, horizons=[128, 64, 32, 16, 8, 4, 2]
) -> None:
    """
    Executes the model training and evaluation loop.

    Parameters:
    model (Any): The model to be trained and evaluated.
    config (py_utils.NestedMap): Configuration object containing model settings and hyperparameters.
    workdir (str): Working directory for saving logs and checkpoints.
    num_classes (int, optional): Number of classes for classification. Defaults to 2.
    plus_one (bool, optional): Whether to add one to the batch before logging. Defaults to False.

    Returns:
    TrainState: The final training state after completing the training loop.
    """
    logdir = os.path.join(workdir, 'logs', current_time)
    os.makedirs(logdir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Check and add handlers if not already present
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(os.path.join(logdir, 'logs.log'))
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info('config.batch_size: {}'.format(config.batch_size))

    writer = tf.summary.create_file_writer(logdir)

    logger.info('config.batch_size: {}'.format(config.batch_size))

    if config.batch_size % jax.process_count() > 0:
        raise ValueError('Batch size must be divisible by the number of processes')

    local_batch_size = config.batch_size // jax.process_count()
    logger.info('local_batch_size: {}'.format(local_batch_size))
    logger.info('jax.local_device_count: {}'.format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError('Local batch size must be divisible by the number of local devices')

    train_loader, eval_loader, train_size = preprocess_csv(config.dataset_path, batch_size=config.batch_size,)
    steps_per_epoch = train_size // config.batch_size + 1

    tfm = model
    model = pax_fiddle.Config(
        PatchedDecoderFinetuneFinance,
        name='patched_decoder_finetune',
        core_layer_tpl=model.model_p,
    )

    task_p = tasks_lib.SingleTask(
        name='ts-learn',
        model=model,
        train=tasks_lib.SingleTask.Train(
            learner=build_learner(
                learning_rate=config.learning_rate,
                momentum=config.momentum,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.num_epochs,
                steps_per_epoch=steps_per_epoch),
        ),
    )

    # task_p.model.ici_mesh_shape = [1, 1, 1] #TODO: optimize this for parallelization
    # task_p.model.mesh_axis_names = ['replica', 'data', 'mdl']

    # DEVICES = np.array(jax.devices()).reshape([1, 1, 1])
    # MESH = jax.sharding.Mesh(DEVICES, ['replica', 'data', 'mdl'])

    num_devices = jax.local_device_count()
    logger.info(f'num_devices: {num_devices}')
    logger.info(f'device kind: {jax.local_devices()[0].device_kind}') #this line takes up a lot of time

    jax_task = task_p
    key = jax.random.PRNGKey(seed=config.seed)
    key, init_key = jax.random.split(key)

    init_ts = jnp.ones((num_devices, config.input_len))
    init_touts = jnp.ones((num_devices, config.output_len))
    init_batch = NestedMap(input_ts=init_ts, actual_ts=init_touts)

    jax_model_states, _ = trainer_lib.initialize_model_state(
        jax_task,
        init_key,
        init_batch,
        checkpoint_type=checkpoint_types.CheckpointType.GDA,
    )

    jax_model_states.mdl_vars['params']['core_layer'] = tfm._train_state.mdl_vars['params']
    gc.collect()

    jax_task = task_p

    key, train_key, eval_key = jax.random.split(key, 3)
    train_prng_seed = jax.random.split(train_key, num=num_devices)
    eval_prng_seed = jax.random.split(eval_key, num=num_devices)

    replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)

    

    logger.info('Starting Evaluation')

    for horizon_len in horizons:
        p_eval_step = jax.pmap(functools.partial(eval_step, jax_task=jax_task, store_metrics=True, horizon_len=horizon_len), axis_name='batch')

        eval_losses = []
        conf_matrices = []

        for n_batch, batch in enumerate(eval_loader):
            #prepare data for eval
            batch = jnp.array(batch)
            if plus_one:
                batch += jnp.ones_like(batch)
            batch = jnp.log(batch)
            batch = reshape_batch(batch, num_devices)
            #eval step
            step_fun_out, inputs, targets = p_eval_step(
                replicated_jax_states, eval_prng_seed, batch
            )
            metrics = postprocess_metrics(step_fun_out, inputs, targets)
            conf_matrices.append(metrics['confusion matrix'])
            eval_losses.append(jnp.mean(step_fun_out.loss))

        epoch_eval_loss = np.mean(eval_losses)
        conf_matrix = np.sum(conf_matrices, axis=0)
        acc = get_accuracy(conf_matrix)
        logger.info('Horizon Length {}, Eval Loss: {}, Accuracy: {}'.format(horizon_len, epoch_eval_loss, acc))

        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return 