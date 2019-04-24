from data import load_data
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from functions import model, input_fn_train, input_fn_eval, savePickle, loadPickle

train_x, label_train, test_x, label_test, validation_x, validation_label = load_data()


def runNetwork(batch_size, train_steps):

    my_feature_columns = []

    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    configuration = tf.estimator.RunConfig(save_summary_steps=4,
                                           keep_checkpoint_max=1,
                                           save_checkpoints_steps=5)

    classifier_warm = tf.estimator.Estimator( model_fn = model,
                                         params = { 'features': my_feature_columns,
                                                    'hidden_layers': [800, 800],
                                                     'num_of_classes': 10,},
                                         model_dir='/model',
                                         config = configuration,
                                         warm_start_from='/model')

    classifier = tf.estimator.Estimator(model_fn=model,
                                             params={'features': my_feature_columns,
                                                     'hidden_layers': [800, 800],
                                                     'num_of_classes': 10, },
                                             model_dir='/model',
                                             config=configuration)


    print('training started')

    directory = '/model/model.ckpt-'

    validation_loss = []

    validation_accuracy = []

    step_length = 5

    count = int(train_steps/step_length) +1

    tf.summary.scalar('validation_acc', validation_accuracy)

    tf.summary.scalar('validation_loss', validation_loss)
    
    for step in range (1, count):

        if(step==1):
            classifier.train(input_fn=lambda: input_fn_train(train_x, label_train, batch_size),
                     steps=step_length)
        else:
            classifier_warm.train(input_fn=lambda: input_fn_train(train_x, label_train, batch_size),
                     steps=step_length)

        print('Evaluation started')

        path = directory + str(step*step_length)
        print(path)
        print(step)

        if(step==1):
            eval_result = classifier.evaluate(input_fn=lambda: input_fn_eval(validation_x, validation_label, batch_size)
            ,checkpoint_path=path)

        else:
            eval_result = classifier_warm.evaluate(input_fn=lambda: input_fn_eval(validation_x, validation_label, batch_size)
                                              , checkpoint_path=path)

        print(eval_result)

        validation_loss.append(eval_result['loss'])

        validation_accuracy.append(eval_result['accuracy'])

        print('\nvalidation set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    path = directory + str((count-1) * step_length)

    print(validation_loss)

    print(validation_accuracy)

    savePickle(validation_accuracy, validation_loss, '/model/data.pkl')

    showModelPerformance(classifier_warm, path)


def showModelPerformance(classifier_warm, path ):

    eval_accuracy, eval_loss = loadPickle('/model/data.pkl')

    eval_result = classifier_warm.evaluate(input_fn=lambda:
                    input_fn_eval(test_x, label_test, batch_size),checkpoint_path=path)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    print(eval_accuracy)

    print(eval_loss)

    plt.plot(eval_accuracy)

    plt.ylabel('Accuracy')

    plt.title('Validation')

    plt.show()

    plt.plot(eval_loss)

    plt.ylabel('Loss')

    plt.title('Validation')

    plt.show()

    predictions = classifier_warm.predict(input_fn=lambda:

    input_fn_eval(test_x, labels=None, batch_size=batch_size), checkpoint_path=path)

    pred = []

    for token in predictions:
        pred.append(token['class_ids'][0])

    test_confusion = confusion_matrix(label_test, pred)

    print('test Confusion is: ')

    print(test_confusion)

    predictions = classifier_warm.predict(input_fn=lambda:
    input_fn_eval(train_x, labels=None, batch_size=batch_size), checkpoint_path=path)

    pred = []

    for token in predictions:
        pred.append(token['class_ids'][0])

    train_confusion = confusion_matrix(label_train, pred)

    print('train Confusion is: ')

    print(train_confusion)

    return


def evalModel(path):

    my_feature_columns = []

    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    configuration = tf.estimator.RunConfig(save_summary_steps=4,
                                           keep_checkpoint_max=1,
                                           save_checkpoints_steps=5)

    classifier = tf.estimator.Estimator(model_fn=model,
                                        params={'features': my_feature_columns,
                                                'hidden_layers': [50],
                                                'num_of_classes': 10, },
                                        config=configuration)

    eval_result = classifier.evaluate(input_fn=lambda: input_fn_eval(validation_x, validation_label, batch_size)
                                      , checkpoint_path=path)
    print(eval_result)

    return


def runFromModel(batch_size, train_steps, beginckpt):

    my_feature_columns = []

    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    configuration = tf.estimator.RunConfig(save_summary_steps=4,
                                           keep_checkpoint_max=1,
                                           save_checkpoints_steps=5)

    classifier_warm = tf.estimator.Estimator(model_fn=model,
                                             params={'features': my_feature_columns,
                                                     'hidden_layers': [500],
                                                     'num_of_classes': 10, },
                                             model_dir='/model',
                                             config=configuration,
                                             warm_start_from='/model')

    print('training started')

    eval_loss = []

    eval_accuracy = []


    tf.summary.scalar('validation_acc', eval_accuracy)
    
    tf.summary.scalar('validation_loss', eval_loss)
    
    step_length = 2
    
    count = int(train_steps / step_length) + 1 + beginckpt

    for step in range((1+beginckpt), count):
    
        classifier_warm.train(input_fn=lambda: input_fn_train(train_x, label_train, batch_size),
                                  steps=step_length)
    
        print('Evaluation started')
    
        path = directory + str(step * step_length)
    
        print(path)
    
        print(step)
    
        eval_result = classifier_warm.evaluate(
                input_fn=lambda: input_fn_eval(validation_x, validation_label, batch_size)
                , checkpoint_path=path)
    
        print(eval_result)
    
        eval_loss.append(eval_result['loss'])
    
        eval_accuracy.append(eval_result['accuracy'])
    
        print('\nvalidation set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    path = directory + str(100)

    showModelPerformance(classifier_warm, path)


batch_size = 200

train_steps = 15

runNetwork(batch_size, train_steps)

runFromModel(batch_size, train_steps, 50)

# evalModel('/model.ckpt-70')
