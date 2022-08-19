from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras import layers, models
from optparse import OptionParser
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

np.random.seed(7)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                 default=None)
    parser.add_option("--date",         dest="date",                              default=None)
    parser.add_option("--inTag",        dest="inTag",                             default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                         default="9x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])


    ############################## Model definition ##############################

    # flooring custom gradient
    @tf.custom_gradient
    def Fgrad(x):
        def fgrad(dy):
            return dy
        return tf.floor(x), fgrad

    inputs = keras.Input(shape = (N, M, 5), name = 'TowerCluster')

    EMcalibrator = models.Sequential()
    EMcalibrator.add( layers.Conv2D(N, (3, 3), activation='relu', input_shape=(N, M, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros') )
    EMcalibrator.add( layers.MaxPooling2D((2, 2)) )
    EMcalibrator.add( layers.Conv2D(N*2, (3, 3), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros') )
    EMcalibrator.add( layers.MaxPooling2D((2, 2)) )
    EMcalibrator.add( layers.Conv2D(N*2, (3, 3), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros') )
    EMcalibrator.add( layers.Flatten() )
    EMcalibrator.add( layers.Dense(N*2, activation='relu') )
    EMcalibrator.add( layers.Dense(1) )
    EMcalibrator.add( layers.Lambda(Fgrad) )

    HADcalibrator = models.Sequential()
    HADcalibrator.add( layers.Conv2D(N, (3, 3), activation='relu', input_shape=(N, M, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros') )
    HADcalibrator.add( layers.MaxPooling2D((2, 2)) )
    HADcalibrator.add( layers.Conv2D(N*2, (3, 3), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros') )
    HADcalibrator.add( layers.MaxPooling2D((2, 2)) )
    HADcalibrator.add( layers.Conv2D(N*2, (3, 3), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros') )
    HADcalibrator.add( layers.Flatten() )
    HADcalibrator.add( layers.Dense(N*2, activation='relu') )
    HADcalibrator.add( layers.Dense(1) )
    HADcalibrator.add( layers.Lambda(Fgrad) )

    EMcalibrator(layers.Lambda(lambda x : tf.gather(x, [0,1,2], axis=3), name="em3d")(inputs))
    HADcalibrator(layers.Lambda(lambda x : tf.gather(x, [0,1,3], axis=3), name="had3d")(inputs))

    outputs = layers.Concatenate(axis=1, name='outputs')([EMcalibrator, HADcalibrator])

    CNNmodel = keras.Model(inputs, outputs, name = 'TauCalibrator')

    def custom_loss(y_true, y_pred):
        y_em = tf.reshape(y_pred[:,0],(-1,1))
        y_had = tf.reshape(y_pred[:,1],(-1,1))

        return tf.nn.l2_loss( (y_true[0]-y_em) / (y_true[0]+0.1) ) + tf.nn.l2_loss( (y_true[1]-y_had) / (y_true[1]+0.1) )

    print(CNNmodel.summary)
    exit()

    CNNmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss, metrics=['RootMeanSquaredError'], run_eagerly=True)


    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag

    # X is (None, N, M, 5)
    #       N runs over eta, M runs over phi
    #       5 features are: Ieta, Iphi, Iem, Ihad, EgIet
    #
    # Y is (None, 3)
    #       3 targets are: hwVisPt, hwVisEmPt, hwVisHadPt

    X = np.load(indir+'/X'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    Y = Y[:,1:3] # keep only hwVisEmPt, hwVisHadPt

    
    ############################## Model training ##############################

    history = CNNmodel.fit(X, Y, epochs=1, batch_size=128, verbose=1, validation_split=0.1)

























