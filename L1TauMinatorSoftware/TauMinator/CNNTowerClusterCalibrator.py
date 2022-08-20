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


    ############################## Model definition ##############################

    # This model calibrates the TowerCluster:
    #    - two CNNs (one for EM and one fro HAD deposit) that produce a (9,9,1) output prediction per deposit type
    #    - both CNNs have yje custom flooring as last step (possibly not neede when moving to QKeras?)
    #    - the custom loss sums the (9,9,1) tensors and  targets the separate EM and HAD deposits and sums the two losses together 

    # flooring custom gradient
    @tf.custom_gradient
    def Fgrad(x):
        def fgrad(dy):
            return dy
        return tf.floor(x), fgrad

    Inputs = keras.Input(shape = (N, M, 5), name='TowerCluster')

    if options.caloClNxM == "9x9":
        EMcalibrator = models.Sequential(name="EMcalibrator")
        EMcalibrator.add( layers.Conv2D(9,  (3, 3), input_shape=(9, 9, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer1") )
        EMcalibrator.add( layers.Conv2D(18, (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer2") )
        EMcalibrator.add( layers.Conv2D(1,  (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer3") )
        EMcalibrator.add( layers.Lambda(Fgrad) )

        HADcalibrator = models.Sequential(name="HADcalibrator")
        HADcalibrator.add( layers.Conv2D(9,  (3, 3), input_shape=(9, 9, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer1") )
        HADcalibrator.add( layers.Conv2D(18, (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer2") )
        HADcalibrator.add( layers.Conv2D(1,  (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer3") )
        HADcalibrator.add( layers.Lambda(Fgrad) )

    elif options.caloClNxM == "7x7":
        EMcalibrator = models.Sequential(name="EMcalibrator")
        EMcalibrator.add( layers.Conv2D(7,  (3, 3), input_shape=(7, 7, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer1") )
        EMcalibrator.add( layers.Conv2D(14, (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer2") )
        EMcalibrator.add( layers.Conv2D(1,  (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer3") )
        EMcalibrator.add( layers.Lambda(Fgrad) )

        HADcalibrator = models.Sequential(name="HADcalibrator")
        HADcalibrator.add( layers.Conv2D(7,  (3, 3), input_shape=(7, 7, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer1") )
        HADcalibrator.add( layers.Conv2D(14, (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer2") )
        HADcalibrator.add( layers.Conv2D(1,  (3, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer3") )
        HADcalibrator.add( layers.Lambda(Fgrad) )

    elif options.caloClNxM == "5x5":
        EMcalibrator = models.Sequential(name="EMcalibrator")
        EMcalibrator.add( layers.Conv2D(5,  (2, 2), input_shape=(5, 5, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer1") )
        EMcalibrator.add( layers.Conv2D(10, (2, 2),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer2") )
        EMcalibrator.add( layers.Conv2D(1,  (2, 2),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer3") )
        EMcalibrator.add( layers.Lambda(Fgrad) )

        HADcalibrator = models.Sequential(name="HADcalibrator")
        HADcalibrator.add( layers.Conv2D(5,  (2, 2), input_shape=(5, 5, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer1") )
        HADcalibrator.add( layers.Conv2D(10, (2, 2),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer2") )
        HADcalibrator.add( layers.Conv2D(1,  (2, 2),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer3") )
        HADcalibrator.add( layers.Lambda(Fgrad) )

    elif options.caloClNxM == "5x9":
        EMcalibrator = models.Sequential(name="EMcalibrator")
        EMcalibrator.add( layers.Conv2D(9,  (2, 3), input_shape=(5, 9, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer1") )
        EMcalibrator.add( layers.Conv2D(18, (2, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer2") )
        EMcalibrator.add( layers.Conv2D(1,  (2, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="EMlayer3") )
        EMcalibrator.add( layers.Lambda(Fgrad) )

        HADcalibrator = models.Sequential(name="HADcalibrator")
        HADcalibrator.add( layers.Conv2D(9,  (2, 3), input_shape=(5, 9, 3), padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer1") )
        HADcalibrator.add( layers.Conv2D(18, (2, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer2") )
        HADcalibrator.add( layers.Conv2D(1,  (2, 3),                        padding="same", activation='relu', kernel_initializer=RN(seed=7), bias_initializer='zeros', name="HADlayer3") )
        HADcalibrator.add( layers.Lambda(Fgrad) )

    else:
        print(' ** ERROR : requested a non-available shape of the TowerClusters')
        print(' ** EXITING!')
        exit()

    EM  = EMcalibrator(layers.Lambda(lambda x : tf.gather(x, [0,1,2], axis=3), name="EM3CH")(Inputs))
    HAD = HADcalibrator(layers.Lambda(lambda x : tf.gather(x, [0,1,3], axis=3), name="HAD3CH")(Inputs))

    EMout  = tf.reshape(tf.reduce_sum(EM),(-1,1))
    HADout = tf.reshape(tf.reduce_sum(HAD),(-1,1)) 

    Outputs = layers.Concatenate(axis=1, name='Outputs')([EMout, HADout])

    CNNmodel = keras.Model(Inputs, Outputs, name='CNNTowerClusterCalibrator')

    def custom_loss(y_true, y_pred):
        y_pred_em  = y_pred[:,0]
        y_pred_had = y_pred[:,1]

        y_true_em  = y_true[0]
        y_true_had = y_true[1]

        return tf.nn.l2_loss( (y_true_em-y_pred_em) / (y_true_em+0.1) ) + tf.nn.l2_loss( (y_true_had-y_pred_had) / (y_true_had+0.1) )

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

    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/CNNTowerClusterCalibrator_plots')

    history = CNNmodel.fit(X, Y, epochs=3, batch_size=12, verbose=1, validation_split=0.1)

    CNNmodel.save(outdir + '/CNNTowerClusterCalibrator')
    EMcalibrator.save(outdir + '/CNNTowerClusterCalibrator/EMcalibrator')
    HADcalibrator.save(outdir + '/CNNTowerClusterCalibrator/HADcalibrator')

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/CNNTowerClusterCalibrator_plots/loss.pdf')
    plt.close()

    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model RootMeanSquaredError')
    plt.ylabel('RootMeanSquaredError')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/CNNTowerClusterCalibrator_plots/RootMeanSquaredError.pdf')
    plt.close()

    print('\nTrained model saved to folder: {}'.format(outdir))

