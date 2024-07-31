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


def random_masking(batch_train, input_len=512, context_len=512, output_len=128):
    """
    Casts a random mask on the training data such that up to input_len items from the back are dropped,
    and the last output_len items are used as output data.

    Parameters:
    batch_train (array-like): The training data batch.
    input_len (int, optional): The length of input sequences from the back that may be dropped. Defaults to 512.
    context_len (int, optional): The context length including the horizon. Defaults to 512.
    output_len (int, optional): The length to be used as output data. Defaults to 128.

    Returns:
    tuple: A tuple containing:
        - input_sequences (jnp.ndarray): The masked input sequences.
        - output_sequences (jnp.ndarray): The sequences used as output (typically the last output_len items).
        - input_padding (jnp.ndarray): The padding mask indicating which input items are kept.
    """
    batch_size, seq_len = batch_train.shape
    random_drop = np.random.randint(0, context_len-output_len) 
    if random_drop > 0:
        batch_train = batch_train[:, :-random_drop]
        prepend = jnp.ones((batch_size, random_drop))
        batch_train = jnp.concatenate([prepend, batch_train], axis=1)
    output_sequences = batch_train[:, -output_len:]
    input_sequences = batch_train[:, :-output_len]

    nums = jnp.arange(0, context_len)
    input_padding = jnp.array(nums).reshape((1, context_len))
    input_padding = jnp.repeat(input_padding, batch_size, axis=0)
    random_indices = np.random.randint(random_drop, context_len-output_len, size=(batch_size, 1))
    random_indices = jnp.repeat(random_indices, context_len, axis=1)
    input_padding = jnp.where(input_padding >= random_indices, 0, 1)

    return input_sequences, output_sequences, input_padding


def train_step(states, prng_key, batch, jax_task=None):
    """
    Performs a single training step for a JAX-based learning model.

    This function prepares the batch data, converts it into the required format,
    and invokes the single learner's training step function from `trainer_lib`.

    Args:
        states: A data structure containing the model's states, which might include
                parameters, optimizer state, and any other stateful components.
        prng_key: A JAX PRNG key, used for random number generation in a safe and
                  reproducible manner.
        batch: A batch of training data that includes both inputs and target sequences.
        jax_task: Specifies the JAX task or model being trained. Defaults to None.

    Returns:
        Updated states after completing the training step, in a tuple (state, step_function_output)
    """
    input_map, output_sequences = prepare_batch_data(batch)
    inputs = NestedMap(input_ts=input_map['input_ts'], input_padding=input_map['input_padding'], actual_ts=output_sequences)
    return trainer_lib.train_step_single_learner(
        jax_task, states, prng_key, inputs
    )

def eval_step(states, prng_key, batch, jax_task=None, store_metrics=False, horizon_len=128):
    """
    Performs a single evaluation step for a JAX-based learning model.

    This function prepares the batch data, converts it into the required format,
    and invokes the single learner's evaluation step function from `trainer_lib`.

    Args:
        states: A data structure containing the model's states, which might include
                parameters, optimizer state, and any other stateful components.
        prng_key: A JAX PRNG key, used for random number generation in a safe and
                  reproducible manner.
        batch: A batch of training data that includes both inputs and target sequences.
        jax_task: Specifies the JAX task or model being trained. Defaults to None.

    Returns:
        A tuple containing:
        - step_function_output (contains the loss here)
        - input sequences
        - output_sequences (ground truth)
    """
    input_map, output_sequences = prepare_batch_data(batch, train=False, horizon_len=horizon_len)
    inputs = NestedMap(input_ts=input_map['input_ts'], actual_ts=output_sequences)
    states = states.to_eval_state()
    _, step_fun_out = trainer_lib.eval_step_single_learner(
        jax_task, states, prng_key, inputs
    )
    return step_fun_out, inputs['input_ts'], output_sequences


def prepare_batch_data(batch, train=True, input_len=512, context_len=512, output_len=128, horizon_len=128):
    """
    Prepares the batch data for training or evaluation by generating input sequences, output sequences, and input padding.

    Parameters:
    batch (array-like): The input data batch.
    train (bool, optional): Flag to indicate whether to prepare the batch for training or evaluation. Defaults to True.
    input_len (int, optional): The length of input sequences to be considered. Defaults to 512.
    context_len (int, optional): The length of the context. Defaults to 512.
    output_len (int, optional): The length of the output sequences. Defaults to 128.
    horizon_len (int, optional): The length of the prediction horizon. Defaults to 128. Must divide sequence_length.

    Returns:
    tuple: A tuple containing:
        - input_map (NestedMap): A mapping of input sequences, input padding, and input frequency.
        - output_sequences (jnp.ndarray): The sequences used as output (typically the last output_len items).
    """
    batch_size, sequence_length = batch.shape
    num_input_patches = sequence_length // input_len + 1

    if train: 
        input_sequences, output_sequences, input_padding = random_masking(batch_train=batch)
    else:
        input_sequences = []
        output_sequences = []
        for input_end in range(context_len, sequence_length, horizon_len):
            input_start = input_end-context_len
            input_sequences.append(batch[:, input_start:input_end])
            output_sequences.append(batch[:, input_end:input_end+horizon_len])
        input_sequences = jnp.concatenate(input_sequences, axis=0)
        output_sequences = jnp.concatenate(output_sequences, axis=0)
        batch_size = input_sequences.shape[0]
        input_padding = jnp.zeros((batch_size, context_len))

    inp_freq = jnp.zeros((batch_size, 1))
    
    input_map = NestedMap({
        "input_ts": input_sequences,
        "input_padding": input_padding,
        "freq": inp_freq
    })

    return input_map, output_sequences

def preprocess_csv(file_path, batch_size=32, train_ratio=0.75):
    """
    Preprocesses a CSV file into training and evaluation TensorFlow datasets.

    Parameters:
    file_path (str): Path to the CSV file.
    batch_size (int, optional): The size of each batch. Defaults to 32.
    train_ratio (float, optional): Ratio of data to be used for training. The rest is used for evaluation. Defaults to 0.75.

    Returns:
    tuple: A tuple containing:
        - train_dataset (tf.data.Dataset): Training dataset.
        - eval_dataset (tf.data.Dataset): Evaluation dataset.
        - train_size (int): Number of training samples.
    """
    df = pd.read_csv(file_path, dtype='float64')
    df = df.dropna(axis=1, how='any')
    df = df.transpose()
     # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate split index
    train_size = int(len(df) * train_ratio)
    train_size -= train_size % batch_size

    # Split into train and eval DataFrames
    train_df = df[:train_size]
    eval_df = df[train_size:]

    # Convert DataFrames to TensorFlow Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_df.values)
    eval_dataset = tf.data.Dataset.from_tensor_slices(eval_df.values)

    # Batch the datasets
    train_dataset = train_dataset.shuffle(buffer_size=batch_size*10).repeat().batch(batch_size)
    eval_dataset = eval_dataset.batch(batch_size)

    return train_dataset, eval_dataset, train_size


def reshape_batch(batch, num_devices):
    """
    Reshapes and pads a batch to evenly distribute data among multiple devices.

    Parameters:
    batch (array-like): The input batch to be reshaped.
    num_devices (int): The number of devices to distribute the batch across.

    Returns:
    array: The reshaped batch.
    """
    batch = jnp.array(batch)
    total_batch_size = batch.shape[0]
    if total_batch_size % num_devices:
        pad_needed = num_devices - total_batch_size % num_devices
        pad_ones = jnp.ones((pad_needed, batch.shape[1]))
        batch = jnp.concatenate([batch, pad_ones], axis=0)
    device_batch_size = batch.shape[0] // num_devices
    batch = batch.reshape((num_devices, device_batch_size, -1))
    return batch


@pax_fiddle.auto_config
def build_learner(learning_rate:float, momentum:float, warmup_epochs:int, total_epochs:int, steps_per_epoch:int) -> learners.Learner:
  return pax_fiddle.Config(
      learners.Learner,
      name='learner',
      loss_name='mse_loss',
      optimizer=optimizers.Sgd(
        clip_gradient_norm_to_value=1.,
        learning_rate=learning_rate,
        momentum=momentum,
        lr_schedule=pax_fiddle.Config(
            schedules.LinearRampupCosineDecay,
            warmup_steps=warmup_epochs*steps_per_epoch,
            decay_start=warmup_epochs*steps_per_epoch,
            decay_end=total_epochs*steps_per_epoch,
            min_ratio=0.,
            max=1. # this value is multiplied by the base learning rate (https://github.com/google/praxis/blob/da4fe8dfc762e510a26c4c241f319b38c1f34366/praxis/optimizers.py#L1194)
        ),
      ),
      # Linear probing i.e we hold the transformer layers fixed. (deactivated)
    #   bprop_variable_exclusion=['.*/stacked_transformer_layer/.*'],
  )


class PatchedDecoderFinetuneFinance(patched_decoder.PatchedDecoderFinetuneModel):
  """Model class for finetuning patched time-series decoder.
  We adjust the default implementation to include masking during fine-tuning 

  For more detailed description of the functions see https://github.com/google-research/timesfm/blob/master/src/timesfm/patched_decoder.py#L490
  """

  def compute_predictions(self, input_batch: NestedMap) -> NestedMap:
    input_ts = input_batch[_INPUT_TS]
    if 'input_padding' in input_batch:
        input_padding = input_batch['input_padding']
    else:
        input_padding = jnp.zeros_like(input_ts)
    context_len = input_ts.shape[1]
    input_patch_len = self.core_layer_tpl.patch_len
    context_pad = (
        (context_len + input_patch_len - 1) // input_patch_len
    ) * input_patch_len - context_len

    input_ts = jnp.pad(input_ts, [(0, 0), (context_pad, 0)])
    input_padding = jnp.pad(
        input_padding, [(0, 0), (context_pad, 0)], constant_values=1
    )
    freq = jnp.ones([input_ts.shape[0], 1], dtype=jnp.int32) * self.freq
    new_input_batch = NestedMap(
        input_ts=input_ts,
        input_padding=input_padding,
        freq=freq,
    )
    return self.core_layer(new_input_batch)

  def compute_loss(
      self, prediction_output: NestedMap, input_batch: NestedMap
  ) -> Tuple[NestedMap, NestedMap]:
    output_ts = prediction_output[_OUTPUT_TS]
    actual_ts = input_batch[_TARGET_FUTURE]
    pred_ts = output_ts[:, -1, 0 : actual_ts.shape[1], :]
    loss = jnp.square(pred_ts[:, :, 0] - actual_ts)
    mse_loss = loss.mean()
    for i, quantile in enumerate(self.core_layer.quantiles):
      loss += self._quantile_loss(pred_ts[:, :, i + 1], actual_ts, quantile)
    loss = loss.mean()
    loss_weight = jnp.array(1.0, dtype=jnp.float32)
    per_example_out = NestedMap(prediction=pred_ts[:, :, 0])
    return {"mse_loss": (mse_loss, loss_weight), "avg_qloss": (loss, loss_weight)}, per_example_out

def postprocess_metrics(step_fun_out, inputs, targets):
    preds = step_fun_out.per_example_out['prediction'][0]
    targets = targets.reshape(-1, targets.shape[-1])
    inputs = inputs.reshape(-1, inputs.shape[-1])
    metrics = {}
    pred_returns = get_returns(preds, inputs)
    target_returns = get_returns(targets, inputs)
    metrics['confusion matrix'] = get_confusion_matrix(pred_returns, target_returns)
    return metrics

def train_and_evaluate(
    model: Any, config: py_utils.NestedMap, workdir: str, num_classes=2, plus_one=False
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

    train_loader, eval_loader, train_size = preprocess_csv(config.dataset_path, batch_size=config.batch_size)
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

    p_train_step = jax.pmap(functools.partial(train_step, jax_task=jax_task), axis_name='batch')
    p_eval_step = jax.pmap(functools.partial(eval_step, jax_task=jax_task, store_metrics=True), axis_name='batch')

    replicated_jax_states = trainer_lib.replicate_model_state(jax_model_states)

    train_losses = []

    checkpoint_path = os.path.join(workdir, 'checkpoints/fine-tuning-' + current_time)

    logger.info('Starting training loop')
    for n_batch, batch in enumerate(train_loader):
        #prepare data for train
        batch = jnp.array(batch)
        if plus_one:
            batch += jnp.ones_like(batch)
        batch = jnp.log(batch)
        batch = reshape_batch(batch, num_devices)
        replicated_jax_states, step_fun_out = p_train_step(
            replicated_jax_states, train_prng_seed, batch
        )
        # print('train loss:', step_fun_out.loss)
        train_losses.append(jnp.mean(step_fun_out.loss))

        if n_batch % steps_per_epoch == 0:
            epoch = n_batch // steps_per_epoch
            if epoch > config.num_epochs:
                break

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

            epoch_train_loss = np.mean(train_losses)
            epoch_eval_loss = np.mean(eval_losses)
            conf_matrix = np.sum(conf_matrices, axis=0)
            acc = get_accuracy(conf_matrix)
            logger.info('Epoch {} \n Train Loss: {}, Eval Loss: {}, Accuracy: {}'.format(epoch, epoch_train_loss, epoch_eval_loss, acc))

            with writer.as_default():
                tf.summary.scalar('Train Loss', epoch_train_loss, step=epoch)
                tf.summary.scalar('Eval Loss', epoch_eval_loss, step=epoch)
                tf.summary.scalar('Accuracy', acc, step=epoch)
            
            train_losses = []

            if epoch>0 and (epoch % config.epochs_per_checkpoint) == 0:
                print("Saving checkpoint.")
                jax_state_for_saving = py_utils.maybe_unreplicate_for_fully_replicated(
                    replicated_jax_states
                )
                checkpoints.save_checkpoint(
                    jax_state_for_saving, checkpoint_path, overwrite=False
                )

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return 