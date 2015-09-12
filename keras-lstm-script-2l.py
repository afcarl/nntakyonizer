from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import numpy as np
import os
import scipy.io.wavfile as wav

def openWavFile(fileName):
    data = wav.read(fileName)
    nparray = data[1].astype('float32') / 32767.0
    return nparray, data[0]
        
def blockToFFT(data):
    retBlock = []
    for blockData in data:
        fftBlock = np.fft.fft(blockData)
        retBlock.append(np.concatenate((np.real(fftBlock), np.imag(fftBlock))))
    return retBlock

def dataToBlock(data, snum, csize):
    block = []
    sampCount = data.shape[0]
    sectSize = int(sampCount/snum)
    blockSize = int(sectSize/csize)
    for z in range(0, snum):
        sectBlock = []
        mydata = data[z * sectSize:(z + 1) * sectSize]
        for x in range(0, csize):
            myBlock = mydata[x * blockSize:(x + 1) * blockSize]
            if myBlock.shape[0] < blockSize:
                padding = np.zeros((blockSize - myBlock.shape[0],))
                myBlock = np.concatenate((myBlock, padding))
            sectBlock.append(myBlock)
        block.append(sectBlock)
    return block

def normalizeBlock(data):
    # xlen = len(data)
    # ylen = len(data[0])
    
    # npNorm = np.zeros((xlen,ylen))
    # for x in xrange(xlen):
        # for y in xrange(ylen):
            # npNorm[x][y] = data[x][y]
    npNorm = np.asarray(data)
    mean = np.mean(np.mean(npNorm, axis=0), axis=0) #Mean across num examples and num timesteps
    std = np.sqrt(np.mean(np.mean(np.abs(npNorm-mean)**2, axis=0), axis=0)) # STD across num examples and num timesteps
    std = np.maximum(1.0e-8, std) #Clamp variance if too tiny
    
    norm = npNorm.copy()
    norm[:] -= mean
    norm[:] /= std
    return norm, mean, std
    
def getFilesBlockData(sectionNumber, blockSize, xfileName, yfileName=False, asFFT=True):
    xdata, xbrate = openWavFile(xfileName)
    if yfileName:
        ydata, ybrate = openWavFile(yfileName)
    else:
        ydata, ybrate = openWavFile(xfileName)
    xdatablock = dataToBlock(xdata, sectionNumber, blockSize)
    ydatablock = dataToBlock(ydata, sectionNumber, blockSize)
    if asFFT:
        X = blockToFFT(xdatablock)
        Y = blockToFFT(ydatablock)
    else:
        X = xdatablock
        Y = ydatablock
    xNorm, xMean, xStd = normalizeBlock(X)
    yNorm, yMean, yStd = normalizeBlock(Y)
    yNormA = yNorm.shape[0]
    yNormB = yNorm.shape[1]
    yNormC = yNorm.shape[2]
    #flatY = np.reshape(yNorm, (yNormA, yNormB * yNormC))
    return xNorm, yNorm
    
def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
    for cur_unit in xrange(num_recurrent_units):
        model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
    model = Sequential()
    #This layer converts frequency space to hidden space
    model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
    for cur_unit in xrange(num_recurrent_units):
        model.add(GRU(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
    #This layer converts hidden space back to frequency space
    model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def trainModel(model, xObj, yObj,trIter, epoch, save=True):
    for x in range(0, trIter):
        model.fit(xObj, yObj, batch_size=1, nb_epoch=epoch, verbose=1, validation_split=0.0)
        
def saveModelWeights(model, networkName):
    model.save_weights(networkName + '-weights')
    
sections = 16
section_blocks = 512
#lstmNetwork = Network('takyonNetwork', 1024, 'takyon-inst.wav', yfileName='takyon-vocal.wav')
xblock, yblock = getFilesBlockData(sections, section_blocks, 'takyon-inst-8.wav', yfileName='takyon-vocal-8.wav', asFFT=False)
xx = xblock.shape[0]
xy = xblock.shape[1]
xz = xblock.shape[2]

newYBlock = []
for yx in range(0, xblock.shape[0]):
	tempblock = []
	for yy in range(0, xblock.shape[1]):
		for yz in range(0, xblock.shape[2]):
			tempblock.append(yblock[yx][yy][yz])
	newYBlock.append(tempblock)

yblock = np.asarray(newYBlock)
yblocksize = yblock.shape[1]
print xblock.shape
print yblock.shape
maxlen = int(xblock.shape[2])

embedding_dims = maxlen + 10
conv_dims = embedding_dims + 20

embedding_dims_b = embedding_dims * 2
conv_dims_b = conv_dims * 2

nb_filters = 2
filter_length = 250

nb_filters_b = 1
filter_length_b = 175

batch_size = 1
epochs_per_iter = 10

model = Sequential()
#model.add(Embedding(maxlen, embedding_dims))
model.add(LSTM(maxlen, conv_dims, return_sequences=True))
model.add(Dropout(0.01))
#######################################################################################
model.add(Convolution1D(input_dim=conv_dims, nb_filter=nb_filters, filter_length=filter_length, activation="relu"))
output_size = nb_filters * (section_blocks - filter_length + 1) #6 * ((64 - 32) + 1)/ 2
print output_size
model.add(Flatten())
model.add(Dense(output_size, yblocksize,activation="linear"))
model.add(Reshape(xy / 2, xz * 2))

model.add(LSTM(xz * 2, conv_dims_b * 2, return_sequences=True))
model.add(Dropout(0.01))
#######################################################################################
model.add(Convolution1D(input_dim=conv_dims_b * 2, nb_filter=nb_filters, filter_length=filter_length, activation="relu"))
output_size = nb_filters * (section_blocks - filter_length + 1) #6 * ((64 - 32) + 1)/ 2
print output_size
model.add(Flatten())
model.add(Dense(output_size, yblocksize,activation="linear"))
model.add(Reshape(xy / 4, xz * 4))

model.add(LSTM(xz * 4, conv_dims_b * 4, return_sequences=True))
model.add(Dropout(0.01))
#######################################################################################
model.add(Convolution1D(input_dim=conv_dims_b * 4, nb_filter=nb_filters, filter_length=filter_length, activation="relu"))
output_size = nb_filters * ((xy / 4) - filter_length + 1) #6 * ((64 - 32) + 1)/ 2
print output_size
model.add(Flatten())
model.add(Dense(output_size, yblocksize,activation="linear"))

model.compile(loss='mean_squared_error', optimizer='rmsprop')
for j in range(0, 20):
	model.fit(xblock, yblock, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1, validation_split=0.0)
	model.save_weights('takyonizer3l', overwrite=True)