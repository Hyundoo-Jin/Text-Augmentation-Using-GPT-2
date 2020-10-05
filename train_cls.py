import argparse, json, os, sys, time, tqdm, pickle
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import sentencepiece as spm
from load_dataset import load_dataset, Sampler

import encoder
import sample_cls as sample
import transfer_clstoken as model
# import model
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

class Arguments() :
    def __init__(self, path) :
        with open(path, 'r') as f :
            args = json.load(f)
        for k, v in args.items() :
            if type(v) == str :
                exec("self.{} = '{}'".format(k, v))
            else :
                exec("self.{} = {}".format(k, v))

def maketree(path) :
    try :
        os.makedirs(path)
    except :
        pass

def randomize(context, hparams, p) :
    if p > 0 :
        mask = tf.random.uniform(shape = tf.shape(context)) < p
        noise = tf.random.uniform(shape = tf.shape(context), minval = 0, maxval = hparams.n_vocab, dtype = tf.int32)
        return tf.where(mask, noise, context)
    else :
        return context

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

parameter_path = 'parameter.json'
args = Arguments(parameter_path)
args.run_name = 'no_classification'
args.label_weight = 0.2

sp = spm.SentencePieceProcessor()
sp.load(args.encoder_path)
hparams = model.default_hparams()
with open('hparams.json') as f:
        hparams.override_from_dict(json.load(f))

hparams.n_vocab = hparams.n_vocab + 9   # cls 토큰 수 추가
if args.sample_length > hparams.n_ctx :
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx
        )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

with tf.Session(config = config) as sess :
    context = tf.placeholder(tf.int32, [args.batch_size, None])
    labels = tf.placeholder(tf.int32, [args.batch_size, None])
    context_in = randomize(context, hparams, args.noise)
    context_in = tf.concat([labels, context_in], axis = 1)
    slice_index = tf.shape(labels)[1]
    output = model.model(hparams = hparams, X = context_in)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = context[:, 1:], logits = output['logits'][:, slice_index:-1]
        )
    )
    
    if args.val_every > 0 :
        val_context = tf.placeholder(tf.int32, [args.val_batch_size, None])
        val_output = model.model(hparams = hparams, X = val_context)
        val_loss = tf.redeuce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = val_context[:, 1:], logits = val_output['logits'][:, :-1]
            )
        )
        val_loss_summary = tf.summary.scalar('val_loss', val_loss)

    tf_sample = sample.sample_sequence(
        hparams = hparams,
        length = args.sample_length,
        context = context,
        labels = labels,
        batch_size = args.batch_size,
        temperature = 1.0,
        top_k = args.top_k,
        top_p = args.top_p
    )

    all_vars = [v for v in tf.all_variables() if 'model' in v.name]
    if args.only_train_transformer_layers :
        train_vars = [v for v in all_vars if '/h' in v.name]
    else :
        train_vars = all_vars

    if args.optimizer == 'adam' :
        opt = tf.train.AdamOptimizer(learning_rate = args.learning_rate)
    elif args.optimizer == 'sgd' :
        opt = tf.train.GradientDescentOptimizer(learning_rate = args.learning_rate)
    else :
        exit('Bad optimizer : ', args.optimizer)

    if args.accumulate_gradients > 1 :
        if args.memory_saving_gradients:
            exit("Memory saving gradients are not implemented for gradient accumulation yet.")
        opt = AccumulatingOptimizer(
            opt=opt,
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.summary.scalar('loss', opt_apply)
    else:
        if args.memory_saving_gradients:
            opt_grads = memory_saving_gradients.gradients(loss, train_vars)
        else:
            opt_grads = tf.gradients(loss, train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)

    summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
    summaries = tf.summary.merge([summary_lr, summary_loss])
    summary_log = tf.summary.FileWriter(
        os.path.join(CHECKPOINT_DIR, args.run_name)
    )
    summary_log = tf.summary.FileWriter(
        os.path.join(CHECKPOINT_DIR, args.run_name)
    )

    saver = tf.train.Saver(
        var_list = all_vars,
        max_to_keep = 3,
        keep_checkpoint_every_n_hours = 2
    )
    sess.run(tf.global_variables_initializer())
    
    ckpt = tf.train.latest_checkpoint(
        os.path.join(CHECKPOINT_DIR, args.run_name))
    if ckpt :
        saver.restore(sess, ckpt)
    
    
    with open('../merged_data_human_only_cls.pkl', 'rb') as f :
        chunks = pickle.load(f)
    data_sampler = Sampler(chunks, transfer = True, labels=True)
    
    print('dataset has', data_sampler.total_size, 'tokens')

    print('Training...')

    # if args.val_every > 0:
    #     # Sample from validation set once with fixed seed to make
    #     # it deterministic during training as well as across runs.
    #     val_data_sampler = Sampler(val_chunks, seed=1)
    #     val_batches = [[val_data_sampler.sample(1024) for _ in range(args.val_batch_size)]
    #                    for _ in range(args.val_batch_count)]

    counter = 1
    counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')

    if os.path.exists(counter_path):
        # Load the step number if we're resuming a run
        # Add 1 so we don't immediately try to save again
        with open(counter_path, 'r') as fp:
            counter = int(fp.read()) + 1

    def save():
        maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
        print(
            'Saving',
            os.path.join(CHECKPOINT_DIR, args.run_name,
                         'model-{}').format(counter))
        saver.save(
            sess,
            os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
            global_step=counter)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter) + '\n')

    def generate_samples(enc, label = False):
        print('Generating samples...')
        context_tokens, context_labels = data_sampler.sample(10, generate = True)
        print('Seed text : {}'.format(enc.DecodeIds(context_tokens.tolist())))
        all_text = []
        index = 0
        while index < args.sample_num :   
            out = sess.run(
                tf_sample,
                feed_dict={context: args.batch_size * [context_tokens],
                           labels : args.batch_size * [context_labels]
                          })
            out = out.tolist()
            for i in range(min(args.sample_num - index, args.batch_size)):
                tokens = out[i]
                tokens = [idx for idx in tokens if idx not in [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]]
                text = enc.DecodeIds(tokens)
                all_text.append(text)
                index += 1
        maketree(os.path.join(SAMPLE_DIR, args.run_name))
        with open(
                os.path.join(SAMPLE_DIR, args.run_name,
                             'samples-{}').format(counter), 'w') as fp:
            fp.write('\n'.join(all_text))
        print(all_text)

    def sample_batch():
        return [data_sampler.sample(args.n_ctx) for _ in range(args.batch_size)]


    avg_loss = (0.0, 0.0)
    start_time = time.time()

    try:
        while True:
            if counter % args.save_every == 0:
                save()
            if counter % args.sample_every == 0:
                generate_samples(sp)
            # if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
            #     validation()

            if args.accumulate_gradients > 1:
                sess.run(opt_reset)
                for _ in range(args.accumulate_gradients):
                    sess.run(
                        opt_compute, feed_dict={context: sample_batch()})
                (v_loss, v_summary) = sess.run((opt_apply, summaries))
            else:
                batch = data_sampler.sample(args.n_ctx)
                context_batch = batch[0].reshape(args.batch_size, -1)
                label_batch = batch[1].reshape(args.batch_size, -1)
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summaries),
                    feed_dict={context: context_batch,
                              labels : label_batch})

            summary_log.add_summary(v_summary, counter)
            avg_loss = (avg_loss[0] * 0.99 + v_loss,
                        avg_loss[1] * 0.99 + 1.0)

            print(
                '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_loss,
                    avg=avg_loss[0] / avg_loss[1]))

            counter += 1

    except KeyboardInterrupt:
        print('interrupted')
        save()
    except tf.errors.ResourceExhaustedError :
        err_batch = batch
