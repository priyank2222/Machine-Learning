from __future__ import division
from __future__ import print_function
import math
import argparse
from datetime import datetime
import json
import os

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel

SAMPLES = 1
LOGDIR = './logdir'
WINDOW = 27
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
fname='./data/texts/all_15MIN.txt'
SPLIT = 0.996
# SPLIT = 0.99985

def create_test_dataset(dataset, train_size):
	dataX, dataY = [], []
	for i in range(train_size+1, len(dataset)-1):
		a = dataset[0:i, 0]
		dataX.append(a)
		dataY.append(dataset[i + 1, 0])
	return np.array(dataX), np.array(dataY)

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--text_out_path',
        type=str,
        default=None,
        help='Path to output txt file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=False,
        help='Use fast generation')
    return parser.parse_args()


def write_text(waveform, filename):
    print (waveform)
    text = waveform
    y = []
    for index, item in enumerate(text):
        # y.append(chr(text[index]))
        y.append(text[index])
    print('Prediction is: ', '\n'.join(str(e) for e in y))
    y = np.array(y)
    np.savetxt(filename, y.reshape(1, y.shape[0]), delimiter="\n", newline="\n", fmt="%s")
    print('Updated text file at {}'.format(filename))


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'])

    samples = tf.placeholder(tf.int32)

    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples)
    else:
        next_sample = net.predict_proba(samples)

    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    # decode = samples
    # Creating a copy of samples
    decode = tf.identity(samples)

    quantization_channels = wavenet_params['quantization_channels']

    with open(fname) as f:
        content = f.readlines()
    waveform = [float(x.replace('\r', '').strip()) for x in content]

    dataframe = pd.read_csv(fname, engine='python')
    dataset = dataframe.values

    # split into train and test sets
    train_size = int(len(dataset) * SPLIT)
    test_size = len(dataset) - train_size

    testX, testY = create_test_dataset(dataset, train_size)

    print (str(len(testX)) + " steps to go...")

    testP = []

    # train, test = waveform[0:train_size,:], waveform[train_size:len(dataset),:]
    # waveform = [144.048970239,143.889691754,143.68922135,143.644718903,143.698498762,143.710396703,143.756327831,143.843187531,143.975287002,143.811912129]

    last_sample_timestamp = datetime.now()
    for step in range(len(testX)):
    # for step in range(2):
        # if len(waveform) > args.window:
        #     window = waveform[-args.window:]
        # else:

        window = testX[step]
        outputs = [next_sample]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window})[0]
        # sample = np.random.choice(np.arange(quantization_channels), p=prediction)
        sample = np.arange(quantization_channels)[np.argmax(prediction)]
        # waveform.append(sample)
        testP.append(sample)
        print (step, sample)

    # Introduce a newline to clear the carriage return from the progress.
    testPredict = np.array(testP).reshape((-1, 1))

    # Save the result as a wav file.
    # if args.text_out_path:
    #     out = sess.run(decode, feed_dict={samples: waveform})
    #     write_text(out, args.text_out_path)
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    testPredictPlot = np.empty_like(dataset, dtype=float)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[train_size+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(testPredictPlot)
    plt.show()

    print('Finished generating.')


if __name__ == '__main__':
    main()
