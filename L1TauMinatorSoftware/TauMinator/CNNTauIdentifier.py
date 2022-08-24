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
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
    positions = keras.Input(shape = 107, name='TowerClusterPosition')
    CNN = models.Sequential(name="CNNidentifier")
    DNN = models.Sequential(name="DNNidentifier")

    if options.caloClNxM == "9x9":
        CNN.add( layers.Conv2D(9, (2, 2), activation='relu', input_shape=(9, 9, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(18, (2, 2), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, activation='relu', name="DNNlayer1") )
        DNN.add( layers.Dense(16, activation='relu', name="DNNlayer2") )
        DNN.add( layers.Dense(1, name="DNNout") )

    elif options.caloClNxM == "7x7":
        CNN.add( layers.Conv2D(7, (2, 2), activation='relu', input_shape=(7, 7, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(14, (2, 2), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, activation='relu', name="DNNlayer1") )
        DNN.add( layers.Dense(16, activation='relu', name="DNNlayer2") )
        DNN.add( layers.Dense(1, name="DNNout") )

    elif options.caloClNxM == "5x5":
        CNN.add( layers.Conv2D(5, (2, 2), activation='relu', input_shape=(5, 5, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(10, (2, 2), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, activation='relu', name="DNNlayer1") )
        DNN.add( layers.Dense(16, activation='relu', name="DNNlayer2") )
        DNN.add( layers.Dense(1, name="DNNout") )

    elif options.caloClNxM == "5x9":
        CNN.add( layers.Conv2D(9, (2, 2), activation='relu', input_shape=(5, 9, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(18, (2, 2), activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, activation='relu', name="DNNlayer1") )
        DNN.add( layers.Dense(16, activation='relu', name="DNNlayer2") )
        DNN.add( layers.Dense(1, name="DNNout") )

    else:
        print(' ** ERROR : requested a non-available shape of the TowerClusters')
        print(' ** EXITING!')
        exit()

    CNNflatened = CNN(layers.Lambda(lambda x : x, name="CNNlayer0")(images))
    middleMan = layers.Concatenate(axis=1, name='middleMan')([CNNflatened, positions])
    TauIdentified = DNN(layers.Lambda(lambda x : x, name="TauIdentifier")(middleMan))

    TauIdentifierModel = keras.Model([images, positions], TauIdentified, name='TauCNNIdentifier')

    # metrics = [tf.keras.metrics.BinaryAccuracy('BA'), tf.keras.metrics.FalseNegatives(name='FN'), tf.keras.metrics.FalsePositive(name='FP'), tf.keras.metrics.TrueNegatives(name='TN'), tf.keras.metrics.TruePositive(name='TP')]
    TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], run_eagerly=True)

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 107)
    #       107 are the OHE version of the 35 ieta (absolute) and 72 iphi values
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forIdentifier.npz')['arr_0']


    ############################## Model training ##############################

    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauCNNIdentifier_plots')

    history = TauIdentifierModel.fit([X1, X2], Y, epochs=10, batch_size=128, verbose=1, validation_split=0.1)

    TauIdentifierModel.save(outdir + '/TauCNNIdentifier')

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/TauCNNIdentifier_plots/loss.pdf')
    plt.close()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/TauCNNIdentifier_plots/accuracy.pdf')
    plt.close()

    # plt.plot(history.history['FP'])
    # plt.plot(history.history['val_FP'])
    # plt.title('False Positives')
    # plt.ylabel('False Positive Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifier_plots/false_positives.pdf')
    # plt.close()

    # plt.plot(history.history['TP'])
    # plt.plot(history.history['val_TP'])
    # plt.title('True Positives')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifier_plots/true_positives.pdf')
    # plt.close()

    # plt.plot(history.history['FN'])
    # plt.plot(history.history['val_FN'])
    # plt.title('False Negatives')
    # plt.ylabel('False Negative Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifier_plots/false_negatives.pdf')
    # plt.close()

    # plt.plot(history.history['TN'])
    # plt.plot(history.history['val_TN'])
    # plt.title('True Negatives')
    # plt.ylabel('True Negative Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifier_plots/true_negatives.pdf')
    # plt.close()

    print('\nTrained model saved to folder: {}'.format(outdir))

