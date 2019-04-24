import tensorflow as tf
import os
import pickle


def input_fn_train(features, labels, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if (batch_size>0):
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def input_fn_eval(features, labels, batch_size):

    # this check is for prediction which input label is None
    if (labels is None):
        dataset = tf.data.Dataset.from_tensor_slices(dict(features))

    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if (batch_size>0):
        dataset = dataset.batch(batch_size)

    return dataset


def model(features, labels, mode, params ):


    FF_NN = tf.feature_column.input_layer(features, params['features'])

    count= 0

    for layer in params['hidden_layers']:

        count+=1

        name= "layer"+str(count)

        FF_NN = tf.layers.dense(FF_NN, units=layer, activation=tf.nn.relu, name=name)

        weights = tf.get_default_graph().get_tensor_by_name(os.path.split(FF_NN.name)[0] + '/kernel:0')

        tf.summary.histogram((FF_NN.name)[0], weights)

        with tf.variable_scope(name, reuse=True):

            w = tf.get_variable('kernel')

            tf.summary.histogram(name, w)

    logits = tf.layers.dense(FF_NN, params['num_of_classes'], activation=None)

    predicted_classes = tf.argmax(logits, 1)

    pred = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {

            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,

        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # loss = tf.sqrt(tf.reduce_mean(tf.square(logits - labels)))

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')

    metrics = {'accuracy': accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:

        print('here, is training')

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.3)

        tf.summary.scalar('train_loss', loss)

        tf.summary.scalar('train_accuracy', accuracy[1])

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        summary_hook = tf.train.SummarySaverHook(
            2,
            output_dir='/log',
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, predictions=predicted_classes,
                                          eval_metric_ops=metrics,
                                          training_hooks=[summary_hook])
    return


def savePickle(accuracy, loss, filePath):

    with open(filePath, 'wb') as output:
        pickle.dump(accuracy, output)
        pickle.dump(loss, output)

    return


def loadPickle(filePath):

    with open(filePath, 'rb') as input:
        accuracy = pickle.load(input)
        loss = pickle.load(input)
    return accuracy, loss
