from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import numpy as np
import shap
import os

np.random.seed(7)

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)


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

    # This model calibrates the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 
    #    - the custom loss targets the visPt of the tau

    images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
    positions = keras.Input(shape = 107, name='TowerClusterPosition')
    CNN = models.Sequential(name="CNNcalibrator")
    DNN = models.Sequential(name="DNNcalibrator")

    if options.caloClNxM == "9x9":
        CNN.add( layers.Conv2D(9, (2, 2), input_shape=(9, 9, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer1') )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(18, (2, 2), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer3') )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, name="DNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer1') )
        DNN.add( layers.Dense(16, name="DNNlayer2") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer2') )
        DNN.add( layers.Dense(1, name="DNNout") )

    elif options.caloClNxM == "7x7":
        CNN.add( layers.Conv2D(7, (2, 2), input_shape=(7, 7, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer1') )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(14, (2, 2), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer3') )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, name="DNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer1') )
        DNN.add( layers.Dense(16, name="DNNlayer2") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer2') )
        DNN.add( layers.Dense(1, name="DNNout") )

    elif options.caloClNxM == "5x5":
        CNN.add( layers.Conv2D(5, (2, 2), input_shape=(5, 5, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer1') )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(10, (2, 2), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer3') )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, name="DNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer1') )
        DNN.add( layers.Dense(16, name="DNNlayer2") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer2') )
        DNN.add( layers.Dense(1, name="DNNout") )

    elif options.caloClNxM == "5x9":
        CNN.add( layers.Conv2D(9, (2, 2), input_shape=(5, 9, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer1') )
        CNN.add( layers.MaxPooling2D((2, 2), name="CNNlayer2") )
        CNN.add( layers.Conv2D(18, (2, 2), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.Activation('relu', name='reluCNNlayer3') )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, name="DNNlayer1") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer1') )
        DNN.add( layers.Dense(16, name="DNNlayer2") )
        CNN.add( layers.Activation('relu', name='reluDNNlayer2') )
        DNN.add( layers.Dense(1, name="DNNout") )

    else:
        print(' ** ERROR : requested a non-available shape of the TowerClusters')
        print(' ** EXITING!')
        exit()

    CNNflatened = CNN(layers.Lambda(lambda x : x, name="CNNlayer0")(images))
    middleMan = layers.Concatenate(axis=1, name='middleMan')([CNNflatened, positions])
    TauCalibrated = DNN(layers.Lambda(lambda x : x, name="TauCalibrator")(middleMan))

    TauCalibratorModel = keras.Model([images, positions], TauCalibrated, name='TauCNNCalibrator')

    def custom_loss(y_true, y_pred):
        return tf.nn.l2_loss( (y_true - y_pred) / (y_true + 0.1) )

    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
    TauCalibratorModel = prune.prune_low_magnitude(TauCalibratorModel, **pruning_params)

    TauCalibratorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=custom_loss, metrics=['RootMeanSquaredError'], run_eagerly=True)

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibratorPruning'+options.caloClNxM+'Training'+options.inTag

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 107)
    #       107 are the OHE version of the 35 ieta (absolute) and 72 iphi values
    #
    # Y is (None, 4)
    #       target: visPt, visEta, visPhi, DM

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forCalibrator.npz')['arr_0']

    
    ############################## Model training ##############################

    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibratorPruning'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauCNNCalibratorPruning_plots')

    history = TauCalibratorModel.fit([X1, X2], Y[:0], epochs=3, batch_size=128, verbose=1, validation_split=0.1)

    TauCalibratorModel = strip_pruning(TauCalibratorModel)
    TauCalibratorModel.save(outdir + '/TauCNNCalibratorPruning')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/loss.pdf')
    plt.close()

    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model RootMeanSquaredError')
    plt.ylabel('RootMeanSquaredError')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/RootMeanSquaredError.pdf')
    plt.close()

    w = TauCalibratorModel.layers[0].weights[0].numpy()
    h, b = np.histogram(w, bins=100)
    props = dict(boxstyle='square', facecolor='white', edgecolor='black')
    textstr1 = '\n'.join((r'% of zeros = {}'.format(np.sum(w==0)/np.size(w))))
    plt.figure(figsize=(10,10))
    plt.bar(b[:-1], h, width=b[1]-b[0])
    plt.text(w.min()+0.01, h.max()-1, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
    plt.legend(loc = 'upper left')
    plt.grid(linestyle=':')
    plt.semilogy()
    plt.xlabel('Weight value')
    plt.ylabel('Recurrence')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/model_sparsity.pdf')
    plt.close()


    ############################## Model validation ##############################

    X1_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNValidator'+options.caloClNxM+'/X_Calib_CNN_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    X2_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNValidator'+options.caloClNxM+'/X_Calib_Dense_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    Y_valid  = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNValidator'+options.caloClNxM+'/Y_Calib_'+options.caloClNxM+'_forValidator.npz')['arr_0']

    train_calib = TauCalibratorModel.predict([X1, X2])
    valid_calib = TauCalibratorModel.predict([X1_valid, X2_valid])
    
    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1).ravel()
    dfTrain['calib_pt']   = train_calib.ravel()
    dfTrain['gen_pt']     = Y[:0].ravel()
    dfTrain['gen_eta']    = Y[:1].ravel()
    dfTrain['gen_phi']    = Y[:2].ravel()
    dfTrain['gen_dm']     = Y[:3].ravel()

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel()
    dfValid['calib_pt']   = valid_calib.ravel()
    dfValid['gen_pt']     = Y_valid[:0].ravel()
    dfValid['gen_eta']    = Y_valid[:1].ravel()
    dfValid['gen_phi']    = Y_valid[:2].ravel()
    dfValid['gen_dm']     = Y_valid[:3].ravel()

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0,5,0.1), label=r'Uncalibrated response, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Calibrated response, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/responses_comparison.pdf')
    plt.close()

    # PLOTS PER DM
    tmp0 = dfValid[dfValid['gen_dm']==0]
    tmp1 = dfValid[(dfValid['gen_dm']==1) | (dfValid['gen_dm']==2)]
    tmp10 = dfValid[dfValid['gen_dm']==10]
    tmp11 = dfValid[(dfValid['gen_dm']==11) | (dfValid['gen_dm']==12)]
    plt.figure(figsize=(10,10))
    plt.hist(tmp0['uncalib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=r'DM 0, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp0['uncalib_pt']/tmp0['gen_pt']), np.std(tmp0['uncalib_pt']/tmp0['gen_pt'])),      color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['uncalib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=r'DM 1, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp1['uncalib_pt']/tmp1['gen_pt']), np.std(tmp1['uncalib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['uncalib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=r'DM 10, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp10['uncalib_pt']/tmp10['gen_pt']), np.std(tmp10['uncalib_pt']/tmp10['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['uncalib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=r'DM 11, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp11['uncalib_pt']/tmp11['gen_pt']), np.std(tmp11['uncalib_pt']/tmp11['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/uncalibrated_DM_responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(tmp0['calib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=r'DM 0, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp0['calib_pt']/tmp0['gen_pt']), np.std(tmp0['calib_pt']/tmp0['gen_pt'])),      color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['calib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=r'DM 1, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp1['calib_pt']/tmp1['gen_pt']), np.std(tmp1['calib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['calib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=r'DM 10, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp10['calib_pt']/tmp10['gen_pt']), np.std(tmp10['calib_pt']/tmp10['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['calib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=r'DM 11, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(tmp11['calib_pt']/tmp11['gen_pt']), np.std(tmp11['calib_pt']/tmp11['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/uncalibrated_DM_responses_comparison.pdf')
    plt.close()


    # 2D REPOSNSE VS ETA
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'], label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'], label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'|\eta^{Gen \tau}|')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/response_vs_eta_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PHI
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'], label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'], label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'|\phi^{Gen \tau}|')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/response_vs_phi_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PT
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'], label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'], label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'|p_{T}^{Gen \tau}|')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/response_vs_pt_comparison.pdf')
    plt.close()


    ############################## Feature importance ##############################

#    # since we have two inputs we pass a list of inputs to the explainer
#    explainer = shap.GradientExplainer(TauCalibratorModel, [X1, X2])
#
#    # we explain the model's predictions on the first three samples of the test set
#    shap_values = explainer.shap_values([X1_valid[:3], X2_valid[:3]])
#
#    # since the model has 10 outputs we get a list of 10 explanations (one for each output)
#    print(len(shap_values))
#
#    # since the model has 2 inputs we get a list of 2 explanations (one for each input) for each output
#    print(len(shap_values[0]))
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the first input (this is the feed forward input)
#    shap.image_plot([shap_values[i][0] for i in range(len(shap_values))], X1_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNCalibratorPruning_plots/shap1.pdf')
#    plt.close()

