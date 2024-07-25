# This file is deprecated, but written here are some useful code to train 
# using basic Jax/Flax, instead of PaxML and Praxis for which Google
# have yet to provide comprehensive documentation on.
# There are some problems with Orbax vs PaxML checkpoints and RNG key is not well
# propagated, but otherwise you should achieve approximately the same result with this file too.

import logging
import time
from typing import Any
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import base_hyperparams
from praxis import py_utils
from praxis import pytypes
from praxis.layers import normalizations
from praxis.layers import transformers
import praxis.optimizers as optimizers
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from flax import jax_utils
from flax.training import train_state, orbax_utils, checkpoints
import paxml
from tensorflow.keras.callbacks import TensorBoard
import datetime
import orbax
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
from utils import *

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
instantiate = base_hyperparams.instantiate

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

np.random.seed(0)

def get_conf_matrix(predictions, targets, prepend, threshold=0.001, num_classes=2, horizon_len=None, use_diff=False):
    """
    Computes the confusion matrix, predicted returns, and target returns from the given predictions and targets.

    Parameters:
    predictions (array-like): The predicted values from the model.
    targets (array-like): The true target values.
    prepend (array-like): Values to prepend to the predictions and targets for calculating returns.
    threshold (float, optional): The threshold for classifying the returns. Defaults to 0.001.
    num_classes (int, optional): The number of classes for classification. Defaults to 2.

    horizon_len (int, optional): The prediction horizon length. Must be at most the prediction length.
    If None, it is assumed to be predictions.shape[1], the full prediction length. Defaults to None.

    use_diff (bool, optional): Whether to use the difference for the initial prepend values. 
    If true, calculates the accuracy based on pred[-1]-pred[0] vs targets[-1]-targes[0]. 
    If false, use pred[-1]-prepend vs targets[-1]-preprend Defaults to False.

    Returns:
    tuple: A tuple containing:
        - conf_matrix_jax (jnp.ndarray): The confusion matrix.
        - pred_returns (jnp.ndarray): The predicted returns.
        - target_returns (jnp.ndarray): The target returns.
    """
    # up=0, down=1, stay=2
    if horizon_len is None:
        horizon_len = predictions.shape[1]
    if use_diff:
        prepend_pred = predictions[:, :1]
        prepend_targets = targets[:, :1]
    else:
        prepend_pred = prepend_targets = prepend
    predictions = predictions[:, (horizon_len-1)::horizon_len]
    targets = targets[:, (horizon_len-1)::horizon_len]
    pred_returns = jnp.diff(predictions, n=1, prepend=prepend_pred)
    target_returns = jnp.diff(targets, n=1, prepend=prepend_targets)
    shifted_targets = jnp.concatenate([prepend, targets], axis=1)[:, :-1]
    pred_returns /= shifted_targets
    target_returns /= shifted_targets
    # print(predictions.shape, targets.shape, prepend.shape, pred_returns.shape, target_returns.shape)

    pred_returns = jnp.ravel(pred_returns) #TODO: CHANGE THIS TO TAKE STD DEV OVER t
    target_returns = jnp.ravel(target_returns)

    # Function to classify values
    if num_classes==3:
        def classify(value, threshold=threshold):
            return jnp.where(value > threshold, 0, jnp.where(value < -threshold, 1, 2))
    else:
        # assume that num_classes==2
        assert num_classes==2
        # def classify(value, threshold=threshold):
        #     return jnp.where(value > 0, 0, 1)
        def classify(value, threshold=threshold):
            return jnp.where(value > threshold, 0, jnp.where(value < -threshold, 1, 2))

    pred_directions = classify(pred_returns)
    target_directions = classify(target_returns)

    # Confusion matrix implementation using JAX
    def confusion_matrix_jax(target, pred, num_classes=num_classes):
        return jnp.array([
            [(target == i).astype(jnp.int32).dot((pred == j).astype(jnp.int32)) for j in range(num_classes)] 
            for i in range(num_classes)
        ])
    
    conf_matrix_jax = confusion_matrix_jax(target_directions, pred_directions)
    return conf_matrix_jax, pred_returns, target_returns

class TrainState(train_state.TrainState):
    # Initialize any other state variables here, currently not in use.
    pass


def train_step(state, batch, learning_rate_fn, output_len=128, horizon_len=128, context_len=512):
    """
    Perform a single training step.

    Parameters:
    state (object): An object containing the model parameters and optimizer state.
    batch (tuple): A tuple containing the input and output data for the batch.
    learning_rate_fn (callable): A function representing the learning rate schedule.
    output_len (int, optional): The length of the output sequences to be predicted. Defaults to 128.
    horizon_len (int, optional): The length of the horizon for model prediction. Defaults to 128. 
    context_len (int, optional): The maximum length of the context for model input. Defaults to 512.

    Returns:
    tuple: A tuple containing:
        - loss (float): The computed loss for the current training step.
        - state (trainstate object): The updated state with the gradient applied.
    """
    input_map, output_sequences = prepare_batch_data(batch)
    
    def loss_fn(params):
        predictions = state.apply_fn(
            params,
            input_map,
            horizon_len=horizon_len,
            output_patch_len=output_len,
            max_len=context_len,
        )[0]
        loss = mse(predictions, output_sequences)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    clipper = optax.clip_by_global_norm(1.)
    clipped_grads, _ = clipper.update(grads, state)
    state = state.apply_gradients(grads=clipped_grads)
    return loss, state

def eval_step(state, batch, loss_fn=mse, output_len=128, horizon_len=128, context_len=512, store_metrics=False):
    """
    Perform a single evaluation step.

    Parameters:
    state (object): An object containing the model parameters and optimizer state.
    batch (tuple): A tuple containing the input and output data for the batch.
    loss_fn (callable, optional): A function to compute the loss. Defaults to mean squared error (mse).
    output_len (int, optional): The length of the output sequences to be predicted. Defaults to 128.
    horizon_len (int, optional): The length of the horizon for model prediction. Defaults to 128.
    We currently only support horizon_len <= output_len. horizon_len must divide output_len.
    context_len (int, optional): The maximum length of the context for model input. Defaults to 512.
    store_metrics (bool, optional): Flag to indicate whether to store and return additional metrics. Defaults to False.

    Returns:
    tuple: If store_metrics is True, returns a tuple containing:
        - loss (float): The computed loss for the current evaluation step.
        - new_conf_matrix (jnp.ndarray): The confusion matrix.
        - pred_returns (jnp.ndarray): The predicted returns.
        - target_returns (jnp.ndarray): The target returns.
    
    Otherwise, returns:
    - loss (float): The computed loss for the current evaluation step.
    """
    all_preds = []
    all_output_sequences = []

    input_ts = []
    input_freq = []
    input_padding = []
    output_sequences = []

    num_iters = output_len // horizon_len

    for i in range(num_iters):
        input_map, output_seq = prepare_batch_data(batch, train=False, input_len=context_len+horizon_len*i)
        input_ts.append(input_map['input_ts'])
        input_freq.append(input_map['freq'])
        input_padding.append(input_map['input_padding'])
        output_sequences.append(output_seq[:, :horizon_len])

    input_ts = jnp.concatenate(input_ts, axis=0)
    input_freq = jnp.concatenate(input_freq, axis=0)
    input_padding = jnp.concatenate(input_padding, axis=0)
    output_sequences = jnp.concatenate(output_sequences, axis=0)
    input_map = NestedMap({
        "input_ts": input_ts,
        "input_padding": input_padding,
        "freq": input_freq
    })

    predictions = state.apply_fn(
        state.params,
        input_map,
        horizon_len=horizon_len,
        output_patch_len=output_len,
        max_len=context_len,
    )[0]
    predictions = predictions[:, :horizon_len]
    output_sequences = output_sequences[:, :horizon_len]

    loss = loss_fn(predictions, output_sequences)

    if store_metrics:
        new_conf_matrix, pred_returns, target_returns = get_conf_matrix(predictions=predictions, targets=output_sequences,\
            prepend=input_map['input_ts'][:, -1:], horizon_len=horizon_len, use_diff=False)
        
        return loss, new_conf_matrix, pred_returns, target_returns
    else:
        return loss


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
    random_drop = np.random.randint(0, context_len-output_len) #tune the 32 parameter to adjust training loss curve noisiness
    # random_drop = 0 #new to avoid stochasticity (but turns out to be helpful?)
    if random_drop > 0:
        batch_train = batch_train[:, :-random_drop]
        prepend = jnp.ones((batch_size, random_drop))
        batch_train = jnp.concatenate([prepend, batch_train], axis=1)
    output_sequences = batch_train[:, -output_len:]
    input_sequences = batch_train[:, :-output_len]

    nums = jnp.arange(0, context_len+output_len)
    input_padding = jnp.array(nums).reshape((1, context_len+output_len))
    input_padding = jnp.repeat(input_padding, batch_size, axis=0)
    random_indices = np.random.randint(random_drop, context_len-output_len, size=(batch_size, 1))
    random_indices = jnp.repeat(random_indices, context_len+output_len, axis=1)
    input_padding = jnp.where(input_padding >= random_indices, 0, 1)

    return input_sequences, output_sequences, input_padding


def prepare_batch_data(batch, train=True, input_len=512, context_len=512, output_len=128):
    """
    Prepares the batch data for training or evaluation by generating input sequences, output sequences, and input padding.

    Parameters:
    batch (array-like): The input data batch.
    train (bool, optional): Flag to indicate whether to prepare the batch for training or evaluation. Defaults to True.
    input_len (int, optional): The length of input sequences to be considered. Defaults to 512.
    context_len (int, optional): The length of the context. Defaults to 512.
    output_len (int, optional): The length of the output sequences. Defaults to 128.

    Returns:
    tuple: A tuple containing:
        - input_map (NestedMap): A mapping of input sequences, input padding, and input frequency.
        - output_sequences (jnp.ndarray): The sequences used as output (typically the last output_len items).
    """
    batch_size, sequence_length = batch.shape
    num_input_patches = sequence_length // input_len + 1

    if train: #also guarantee that input length is never larger than context length (usually input length is set to context length in training)
        input_sequences, output_sequences, input_padding = random_masking(batch_train=batch)
        
    else:
        input_sequences = batch[:, max(0, input_len - context_len):input_len]
        input_padding = jnp.zeros((batch_size, context_len+output_len))
        output_sequences = batch[:, input_len:input_len+output_len]

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

def restart_state(model, config, learning_rate_fn):
    """
    Restarts the training state with a new optimizer and model parameters.

    Parameters:
    model (object): The model object containing the parameters and apply function.
    config (object): Configuration object with model settings and hyperparameters.
    learning_rate_fn (callable): The learning rate schedule function.

    Returns:
    TrainState: The initialized training state.
    """
    tx = create_optimizer(learning_rate_fn=learning_rate_fn, momentum=config.momentum)
    apply_fn = functools.partial(
        model._model.apply, 
        method=model._model.decode,
        rngs={
                base_layer.PARAMS: model._key1,
                base_layer.RANDOM: model._key2,
            }
    )
    state = TrainState.create(
        apply_fn=apply_fn,
        params=model._train_state.mdl_vars,
        tx=tx
    )
    return state

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

def save_checkpoint(state, save_dir, keep=10, use_paxml=True, model=None):
    """
    Saves a checkpoint of the current training state.

    Parameters:
    state (object): The current training state.
    save_dir (str): Directory to save the checkpoint.
    keep (int, optional): Number of checkpoints to keep. Defaults to 10.
    use_paxml (bool, optional): Whether to use the PaxML library for checkpointing. Defaults to True.
    model (object, optional): The model object, required if `use_paxml` is True. Defaults to None.
    """
    if use_paxml:
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        logging.info('Saving checkpoint step %d.', state.step)
        old_state = model._train_state
        state = old_state.new_state(old_state.step, state.params, [])
        paxml.checkpoints.save_checkpoint(state, save_dir)
    else:
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        logging.info('Saving checkpoint step %d.', step)
        checkpoints.save_checkpoint_multiprocess(save_dir, state, step, keep=keep)

def train_and_evaluate(
    model: Any, config: py_utils.NestedMap, workdir: str, num_classes=2, plus_one=False
) -> TrainState:
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

    writer = tf.summary.create_file_writer(workdir + '/logs/' + current_time)

    logging.info('config.batch_size: {}'.format(config.batch_size))

    if config.batch_size % jax.process_count() > 0:
        raise ValueError('Batch size must be divisible by the number of processes')

    local_batch_size = config.batch_size // jax.process_count()
    logging.info('local_batch_size: {}'.format(local_batch_size))
    logging.info('jax.local_device_count: {}'.format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError('Local batch size must be divisible by the number of local devices')

    rng = jax.random.PRNGKey(config.seed)

    train_loader, eval_loader, train_size = preprocess_csv(config.dataset_path, batch_size=config.batch_size)
    # _, test_loader, _ = preprocess_csv(config.testset, batch_size=config.batch_size, train_ratio=0)
    steps_per_epoch = train_size // config.batch_size + 1

    learning_rate_fn = create_learning_rate_fn(peak_learning_rate=config.learning_rate, steps_per_epoch=steps_per_epoch, \
        num_epochs=config.num_epochs, warmup_epochs=config.warmup_epochs)

    jax.config.update('jax_disable_jit', False)

    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn),
        axis_name='batch',
    )

    p_eval_step = jax.pmap(
        functools.partial(eval_step, output_len=128, store_metrics=True),
         axis_name='batch'
    )

    p_test_step = jax.pmap(
        functools.partial(eval_step, output_len=4),
         axis_name='batch'
    )

    state = restart_state(model=model, config=config, learning_rate_fn=learning_rate_fn)
    state = jax_utils.replicate(state)

    train_losses = []

    checkpoint_path = workdir + '/checkpoints/fine-tuning-'  + current_time 

    for n_batch, batch in enumerate(train_loader):
        batch = jnp.array(batch)
        if plus_one:
            batch += jnp.ones_like(batch)
        batch = jnp.log(batch)
        batch = reshape_batch(batch, model.num_devices)
        loss, state = p_train_step(state, batch)
        mean_loss = jnp.mean(loss)
        train_losses.append(mean_loss)

        if n_batch % steps_per_epoch == 0:
            # End of 1 epoch, do eval
            epoch = n_batch // steps_per_epoch
            if epoch > config.num_epochs:
                break
            eval_losses = []
            eval_losses_original = []
            conf_matrices = []
            model._eval_context = base_layer.JaxContext.HParams(do_eval=True) #turns off things like dropout(or maybe not)
            for n_batch, batch in enumerate(eval_loader):
                batch = jnp.array(batch)
                if plus_one:
                    batch += jnp.ones_like(batch)
                batch_log = jnp.log(batch)
                batch_log = reshape_batch(batch_log, model.num_devices)

                loss, conf_matrix, _, _ = p_eval_step(state, batch_log)
                mean_loss = jnp.mean(loss)
                eval_losses.append(mean_loss)
                conf_matrices.append(conf_matrix)

            model._eval_context = base_layer.JaxContext.HParams(do_eval=False)

            epoch_train_loss = np.mean(train_losses)
            epoch_eval_loss = np.mean(eval_losses)
            epoch_eval_loss_original = np.mean(eval_losses_original)
            conf_matrices = np.array(conf_matrices)
            conf_matrix = np.sum(conf_matrices, axis=(0,1))
            accuracy_val = accuracy(conf_matrix=conf_matrix)

            print('Epoch {} \n Train Loss: {}, Eval Loss: {}, Original Loss: {}, Accuracy: {}'.format(epoch, epoch_train_loss, epoch_eval_loss, epoch_eval_loss_original, accuracy_val))

            with writer.as_default():
                tf.summary.scalar('Train Loss', epoch_train_loss, step=epoch)
                tf.summary.scalar('Eval Loss Log', epoch_eval_loss, step=epoch)
                tf.summary.scalar('Eval Loss Original', epoch_eval_loss_original, step=epoch)
                tf.summary.scalar('Accuracy', accuracy(conf_matrix=conf_matrix), step=epoch)
            
            train_losses = []

            if (epoch > 0) and (epoch % 10 == 0):
                save_checkpoint(state, checkpoint_path, model=model)

    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    jax.config.update('jax_disable_jit', False)

    return state