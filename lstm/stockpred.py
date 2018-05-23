from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy
import lstm, time #helper libraries
import os
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


DATA_DIRECTORY = './all_normallized'
FILE = 'all_5MIN_normalized' #DO NOT PUT .csv here
# NO_OF_RECORDS = 8
look_back = 50
SPLIT = 0.85
EPOCHS = 75
BATCH_SIZE = 1000
LOGDIR_ROOT = './logdir/20170502'

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories():
    logdir_root = LOGDIR_ROOT
    logdir = logdir_root
    restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': logdir_root,
        'restore_from': restore_from
    }

directories = validate_directories()
logdir = directories['logdir']
restore_from = directories['restore_from']

#Step 1 Load Data
# X_train, y_train, X_test, y_test = lstm.load_data(DATA_DIRECTORY + '/' + FILE + '.csv', NO_OF_RECORDS, True)
dataframe = pd.read_csv(DATA_DIRECTORY + '/' + FILE + '.csv', engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * SPLIT)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Step 2 Build Model
model = Sequential()

# model.add(LSTM(input_dim=1,output_dim=50,return_sequences=True))
model.add(LSTM(50, activation='tanh', input_shape=(1, look_back),return_sequences=True))
# model.add(Dropout(0.2))

model.add(LSTM(50, activation='tanh',return_sequences=True))
# model.add(Dropout(0.2))

model.add(LSTM(50, activation='tanh',return_sequences=False))
# model.add(Dropout(0.2))

model.add(Dense(output_dim=1))
# model.add(Activation('linear'))

start = time.time()
# model.compile(loss='mse', optimizer='rmsprop')
model.compile(loss='mean_squared_error', optimizer='adam')


#Step 3 Train the model
filepath = LOGDIR_ROOT+'/all_5MIN_normalized_{epoch:02d}_{val_loss:.8f}.hdf5'
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Save the model as png file
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)

# model.fit(X_train,y_train,batch_size=BATCH_SIZE,nb_epoch=EPOCHS,validation_split=0.05,callbacks=[checkpointer])
model.fit(trainX, trainY,batch_size=BATCH_SIZE,nb_epoch=EPOCHS,validation_split=0.05,callbacks=[checkpointer])


#Step 4 - Plot the predictions!
# predictions = lstm.predict_sequences_multiple(model, X_test[:100], NO_OF_RECORDS, NO_OF_RECORDS)
trainPredict = model.predict(trainX)
testX2 = numpy.copy(testX)

for i in range(testX.shape[0]):
	testPredOne = model.predict(numpy.array([testX2[i,:,:]]))
	temp = numpy.delete(testX2[i:,:,:], 0, 2)
	testX2[i:,:,:] = numpy.insert(temp, look_back-1, testPredOne[0][0], axis=2)

testPredict = model.predict(testX)
testPredict2 = model.predict(testX2)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testPredict2 = scaler.inverse_transform(testPredict2)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
testScore2 = math.sqrt(mean_squared_error(testY[0], testPredict2[:,0]))
print('Test Score: %.2f RMSE' % (testScore2))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# shift test predictions for plotting
testPredictPlot2 = numpy.empty_like(dataset)
testPredictPlot2[:, :] = numpy.nan
testPredictPlot2[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict2

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(testPredictPlot2)
plt.show()
# lstm.plot_results_multiple(predictions, y_test[:100], NO_OF_RECORDS)
