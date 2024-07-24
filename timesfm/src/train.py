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

NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
instantiate = base_hyperparams.instantiate

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

np.random.seed(0)

def create_learning_rate_fn(
    peak_learning_rate: float,
    steps_per_epoch: int,
    num_epochs: int = 100,
    warmup_epochs: int = 25,
):
    """Create learning rate schedule."""
    print(steps_per_epoch, num_epochs, warmup_epochs)
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=peak_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )
    return schedule_fn

def mse(predictions, targets):
    squared_error = optax.losses.squared_error(predictions=predictions, targets=targets) 
    return jnp.mean(squared_error)

def percent_mse(predictions, targets):
    squared_error = optax.losses.squared_error(predictions=predictions, targets=targets) / targets**2
    return jnp.mean(squared_error)

def accuracy(conf_matrix):
    correct = np.trace(conf_matrix)
    total = np.sum(conf_matrix)
    return correct/total

def chance_rate(conf_matrix):
    total = np.sum(conf_matrix)
    cat = np.sum(conf_matrix, axis=1)
    prob = cat/total
    return np.sum(prob**2)

def get_conf_matrix(predictions, targets, prepend, threshold=0.001, num_classes=2, output_len=None, use_diff=False):
    # up=0, down=1, stay=2
    if output_len is None:
        output_len = predictions.shape[1]
    if use_diff:
        prepend_pred = predictions[:, :1]
        prepend_targets = targets[:, :1]
    else:
        prepend_pred = prepend_targets = prepend
    predictions = predictions[:, (output_len-1)::output_len]
    targets = targets[:, (output_len-1)::output_len]
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
    pass


def train_step(state, batch, learning_rate_fn, output_len=128, horizon_len=128, context_len=512):
    """Perform a single training step."""
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

def eval_step(state, batch, loss_fn=mse, output_len=128, horizon_len=128, context_len=512, store_metrics=False, num_iters=None, use_diff=False):
    if num_iters is None:
        num_iters = horizon_len // output_len + horizon_len % output_len

    all_preds = []
    all_output_sequences = []

    input_ts = []
    input_freq = []
    input_padding = []
    output_sequences = []

    for i in range(num_iters):
        input_map, output_seq = prepare_batch_data(batch, train=False, input_len=context_len+output_len*i)
        input_ts.append(input_map['input_ts'])
        input_freq.append(input_map['freq'])
        input_padding.append(input_map['input_padding'])
        output_sequences.append(output_seq[:, :output_len])

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
    predictions = predictions[:, :output_len]
    output_sequences = output_sequences[:, :output_len]

    loss = loss_fn(predictions, output_sequences)

    if store_metrics:
        new_conf_matrix, pred_returns, target_returns = get_conf_matrix(predictions=predictions, targets=output_sequences,\
            prepend=input_map['input_ts'][:, -1:], output_len=output_len, use_diff=use_diff)
        
        return loss, new_conf_matrix, pred_returns, target_returns
    else:
        return loss

def normalize_data(batch):
    """ Normalizes the data such that the mean is 10"""
    batch = 10 * batch / jnp.mean(batch, axis=1, keepdims=True)
    return batch


def random_masking(batch_train, input_len=512, context_len=512, horizon_len=128):
    """Casts a random mask on the train data such that up to input_len many items from the back are dropped,
    the last horizon_len many items are used as output_data"""
    batch_size, seq_len = batch_train.shape
    random_drop = np.random.randint(0, context_len-horizon_len) #tune the 32 parameter to adjust training loss curve noisiness
    # random_drop = 0 #new to avoid stochasticity (but turns out to be helpful?)
    if random_drop > 0:
        batch_train = batch_train[:, :-random_drop]
        prepend = jnp.ones((batch_size, random_drop))
        batch_train = jnp.concatenate([prepend, batch_train], axis=1)
    output_sequences = batch_train[:, -horizon_len:]
    input_sequences = batch_train[:, :-horizon_len]

    nums = jnp.arange(0, context_len+horizon_len)
    input_padding = jnp.array(nums).reshape((1, context_len+horizon_len))
    input_padding = jnp.repeat(input_padding, batch_size, axis=0)
    random_indices = np.random.randint(random_drop, context_len-horizon_len, size=(batch_size, 1))
    random_indices = jnp.repeat(random_indices, context_len+horizon_len, axis=1)
    input_padding = jnp.where(input_padding >= random_indices, 0, 1)

    return input_sequences, output_sequences, input_padding


def prepare_batch_data(batch, train=True, input_len=512, context_len=512, horizon_len=128):
    batch_size, sequence_length = batch.shape
    num_input_patches = sequence_length // input_len + 1

    if train: #also guarantee that input length is never larger than context length (usually input length is set to context length in training)
        input_sequences, output_sequences, input_padding = random_masking(batch_train=batch)
        
    else:
        input_sequences = batch[:, max(0, input_len - context_len):input_len]
        input_padding = jnp.zeros((batch_size, context_len+horizon_len))
        output_sequences = batch[:, input_len:input_len+horizon_len]

    inp_freq = jnp.zeros((batch_size, 1))
    
    input_map = NestedMap({
        "input_ts": input_sequences,
        "input_padding": input_padding,
        "freq": inp_freq
    })

    return input_map, output_sequences

def preprocess_csv(file_path, batch_size=32, train_ratio=0.75):
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

def create_optimizer(learning_rate_fn, momentum):
    return optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=momentum,
    )

def restart_state(model, config, learning_rate_fn):
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
    """Reshapes and pads batch to send to each device"""
    batch = jnp.array(batch)
    total_batch_size = batch.shape[0]
    if total_batch_size % num_devices:
        pad_needed = num_devices - total_batch_size % num_devices
        pad_ones = jnp.ones((pad_needed, batch.shape[1]))
        batch = jnp.concatenate([batch, pad_ones], axis=0)
    device_batch_size = batch.shape[0] // num_devices
    batch = batch.reshape((num_devices, device_batch_size, -1))
    return batch

def train_and_evaluate(
    model: Any, config: py_utils.NestedMap, workdir: str, num_classes=2, plus_one=False
) -> TrainState:
    """Execute model training and evaluation loop."""

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

def save_checkpoint(state, save_dir, keep=10, use_paxml=True, model=None):
    #this is the pax implementation
    if use_paxml:
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        logging.info('Saving checkpoint step %d.', state.step)
        old_state = model._train_state
        state = old_state.new_state(old_state.step, state.params, [])
        paxml.checkpoints.save_checkpoint(state, save_dir)
    else:
        # below is the flax implementation
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        logging.info('Saving checkpoint step %d.', step)
        checkpoints.save_checkpoint_multiprocess(save_dir, state, step, keep=keep)


def restore_checkpoint(state, save_dir, use_paxml=False):
    #this is the pax implementation
    if use_paxml:
        return checkpoints.restore_checkpoint(state, save_dir)
    #below is the flax implementation
    else:
        return checkpoints.restore_checkpoint(save_dir, state)

def restore_and_evaluate(
    model: Any, config: py_utils.NestedMap, workdir: str, checkpoint_path: str=None, num_classes=2, output_lens=[128, 64, 32, 16, 8, 4, 2], plus_one: bool=True
) -> TrainState:
    """Execute model training and evaluation loop."""

    writer = tf.summary.create_file_writer(workdir + '/logs/' + current_time + '-' + config.dataset.name)

    logging.info('config.batch_size: {}'.format(config.batch_size))

    if config.batch_size % jax.process_count() > 0:
        raise ValueError('Batch size must be divisible by the number of processes')

    local_batch_size = config.batch_size // jax.process_count()
    logging.info('local_batch_size: {}'.format(local_batch_size))
    logging.info('jax.local_device_count: {}'.format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError('Local batch size must be divisible by the number of local devices')

    rng = jax.random.PRNGKey(config.seed)

    train_loader, eval_loader, train_size = preprocess_csv(config.dataset_path, batch_size=config.batch_size, train_ratio=0)
    steps_per_epoch = train_size // config.batch_size + 1

    output_lens = np.array([output_lens])
    output_lens = np.ravel(output_lens)

    state = restart_state(model=model, config=config, learning_rate_fn=0) # set learning rate to 0 in eval
    if checkpoint_path is not None:
        state = restore_checkpoint(state, checkpoint_path)
    state = jax_utils.replicate(state)

    for output_len in output_lens:
        print('output_len:', output_len)
        p_eval_step = jax.pmap(
            functools.partial(eval_step, output_len=output_len, use_diff=True, store_metrics=True),
            axis_name='batch'
        )

        train_losses = []
        eval_losses = []
        conf_matrices = []
        pred_returns_arr = jnp.array([])
        target_returns_arr = jnp.array([])

        model._eval_context = base_layer.JaxContext.HParams(do_eval=True)
        for n_batch, batch in enumerate(eval_loader):
            batch = jnp.array(batch)
            if plus_one:
                batch = jnp.ones_like(batch) + batch
            batch_log = jnp.log(batch)
            batch_log = reshape_batch(batch_log, model.num_devices)

            loss, conf_matrix, pred_returns, target_returns = p_eval_step(state, batch_log)

            pred_returns = jnp.ravel(pred_returns) # actually this is redundant for now: it reshapes (model.num_devices, n) to the same thing
            target_returns = jnp.ravel(target_returns) 

            mean_loss = jnp.mean(loss)
            eval_losses.append(mean_loss)
            conf_matrices.append(conf_matrix)
            pred_returns_arr = jnp.append(pred_returns_arr, pred_returns)
            target_returns_arr = jnp.append(target_returns_arr, target_returns)

        model._eval_context = base_layer.JaxContext.HParams(do_eval=False)
        eval_loss = np.mean(eval_losses)
        conf_matrices = np.array(conf_matrices)
        conf_matrix = np.sum(conf_matrices, axis=(0,1))
        conf_matrices = np.sum(conf_matrices, axis=1)
        accuracies = [accuracy(conf_matrix=conf_matrix) for conf_matrix in conf_matrices]
        acc = np.mean(accuracies)
        acc_std = np.std(accuracies)
        returns_arr = jnp.concatenate([target_returns_arr[:, None], pred_returns_arr[:, None]], axis=1)
        corr = spearmanr(returns_arr)
        returns_arr = pd.DataFrame(returns_arr)
        returns_arr.to_csv(workdir + f'/test_results/returns-{output_len}.csv', index=False)

        print('Eval Loss: ', eval_loss)
        print('Confusion Matrix:', conf_matrix)
        print('Accuracy: ', acc)
        print('Accuracy Std Dev: ', acc_std)
        print('Accuracies:', accuracies)
        print('Spearman Rank Correlation:', corr)
        with writer.as_default():
            tf.summary.scalar('Eval Loss', eval_loss, step=output_len)
            tf.summary.scalar('Accuracy', accuracy(conf_matrix=conf_matrix), step=output_len)

        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state