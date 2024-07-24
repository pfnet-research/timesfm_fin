import numpy as np
import optax
import jax
import jax.numpy as jnp

def create_learning_rate_fn(
    peak_learning_rate: float,
    steps_per_epoch: int,
    num_epochs: int = 100,
    warmup_epochs: int = 25,
):
    """
    Creates a learning rate schedule with a warmup phase followed by a cosine decay phase.

    Parameters:
    peak_learning_rate (float): The peak learning rate to be achieved after the warmup phase.
    steps_per_epoch (int): The number of steps per epoch.
    num_epochs (int, optional): The total number of epochs for training. Defaults to 100.
    warmup_epochs (int, optional): The number of epochs for the warmup phase. Defaults to 25.

    Returns:
    schedule_fn: A function representing the learning rate schedule over the training period.
    """
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

def create_optimizer(learning_rate_fn, momentum):
    """
    Creates an SGD optimizer with momentum using the provided learning rate schedule.

    Parameters:
    learning_rate_fn (function): A function representing the learning rate schedule over time.
    momentum (float): The momentum factor for the SGD optimizer.

    Returns:
    optax.GradientTransformation: The SGD optimizer with the specified learning rate schedule and momentum.
    """
    return optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=momentum,
    )

def mse(predictions, targets):
    """
    Computes the Mean Squared Error (MSE) between the predictions and the targets.

    Parameters:
    predictions (array-like): Predicted values.
    targets (array-like): Ground truth values.

    Returns:
    float: The mean squared error.
    """
    squared_error = optax.losses.squared_error(predictions=predictions, targets=targets) 
    return jnp.mean(squared_error)

def percent_mse(predictions, targets):
    """
    Computes the Mean Squared Error (MSE) as a percentage of the target values.

    Parameters:
    predictions (array-like): Predicted values.
    targets (array-like): Ground truth values.

    Returns:
    float: The percentage mean squared error.
    """
    squared_error = optax.losses.squared_error(predictions=predictions, targets=targets) / targets**2
    return jnp.mean(squared_error)

def accuracy(conf_matrix):
    """
    Computes the classification accuracy from the confusion matrix.

    Parameters:
    conf_matrix (array-like): Confusion matrix.

    Returns:
    float: The accuracy of the classification.
    """
    correct = np.trace(conf_matrix)
    total = np.sum(conf_matrix)
    return correct / total

def chance_rate(conf_matrix):
    """
    Computes the chance rate of classification from the confusion matrix.

    Parameters:
    conf_matrix (array-like): Confusion matrix.

    Returns:
    float: The chance rate of classification.
    """
    total = np.sum(conf_matrix)
    cat = np.sum(conf_matrix, axis=1)
    prob = cat / total
    return np.sum(prob ** 2)