from train import * 

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