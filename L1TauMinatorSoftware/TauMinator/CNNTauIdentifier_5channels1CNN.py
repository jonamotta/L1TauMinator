from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
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


    ############################## Models definition ##############################

    # This model identifies the tau object:
    #    - one CNNs that produce a single output for classification of tau vs anything else

    CNNmodel = models.Sequential(name="CNNTauIdentifier")
    
    if options.caloClNxM == "9x9":
        CNNmodel.add( layers.Conv2D(9, (3, 3), activation='relu', input_shape=(9, 9, 5), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer1") )
        CNNmodel.add( layers.MaxPooling2D((2, 2), name="IDlayer2") )
        CNNmodel.add( layers.Conv2D(18, (3, 3), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer3") )
        CNNmodel.add( layers.Flatten(name="IDlayer4") )
        CNNmodel.add( layers.Dense(18, activation='relu', name="IDlayer5") )
        CNNmodel.add( layers.Dense(1, name="IDout") )

    elif options.caloClNxM == "7x7":
        CNNmodel.add( layers.Conv2D(7, (2, 2), activation='relu', input_shape=(7, 7, 5), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer1") )
        CNNmodel.add( layers.MaxPooling2D((1, 1), name="IDlayer2") )
        CNNmodel.add( layers.Conv2D(14, (2, 2), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer3") )
        CNNmodel.add( layers.Flatten(name="IDlayer4") )
        CNNmodel.add( layers.Dense(14, activation='relu', name="IDlayer5") )
        CNNmodel.add( layers.Dense(1, name="IDout") )

    elif options.caloClNxM == "5x5":
        CNNmodel.add( layers.Conv2D(5, (1, 1), activation='relu', input_shape=(5, 5, 5), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer1") )
        CNNmodel.add( layers.MaxPooling2D((1, 1), name="IDlayer2") )
        CNNmodel.add( layers.Conv2D(10, (1, 1), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer3") )
        CNNmodel.add( layers.Flatten(name="IDlayer4") )
        CNNmodel.add( layers.Dense(10, activation='relu', name="IDlayer5") )
        CNNmodel.add( layers.Dense(1, name="IDout") )

    elif options.caloClNxM == "5x9":
        CNNmodel.add( layers.Conv2D(9, (1, 3), activation='relu', input_shape=(5, 9, 5), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer1") )
        CNNmodel.add( layers.MaxPooling2D((1, 2), name="IDlayer2") )
        CNNmodel.add( layers.Conv2D(18, (1, 3), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="IDlayer3") )
        CNNmodel.add( layers.Flatten(name="IDlayer4") )
        CNNmodel.add( layers.Dense(18, activation='relu', name="IDlayer5") )
        CNNmodel.add( layers.Dense(1, name="IDout") )

    else:
        print(' ** ERROR : requested a non-available shape of the TowerClusters')
        print(' ** EXITING!')
        exit()

    CNNmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], run_eagerly=True)

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag

    # X is (None, N, M, 5)
    #       N runs over eta, M runs over phi
    #       5 features are: Ieta, Iphi, Iem, Ihad, EgIet
    #
    # Y is (None, 1)
    #       1 target is: sgn-bkg ID

    X = np.load(indir+'/X'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forIdentifier.npz')['arr_0']

    
    ############################## Model training ##############################

    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/CNNTauIdentifier_plots')

    history = CNNmodel.fit(X, Y, epochs=3, batch_size=12, verbose=1, validation_split=0.1)

    CNNmodel.save(outdir + '/CNNTauIdentifier')

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/CNNTauIdentifier_plots/loss.pdf')
    plt.close()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/CNNTauIdentifier_plots/accuracy.pdf')
    plt.close()

    print('\nTrained model saved to folder: {}'.format(outdir))

