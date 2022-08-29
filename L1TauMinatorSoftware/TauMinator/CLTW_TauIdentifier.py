from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import os

np.random.seed(77)

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                      default="9x9")
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauCNNIdentifier_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forIdentifier.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forIdentifier.npz')['arr_0']

    
    ############################## Models definition ##############################

    # This model identifies the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')
        CNN = models.Sequential(name="CNNidentifier")
        DNN = models.Sequential(name="DNNidentifier")

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        CNN.add( layers.Conv2D(16, wndw, input_shape=(N, M, 3), kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer1") )
        CNN.add( layers.BatchNormalization(name='BNlayer1') )
        CNN.add( layers.Activation('relu', name='reluCNNlayer1') )
        CNN.add( layers.MaxPooling2D(wndw, name="CNNlayer2") )
        CNN.add( layers.Conv2D(24, wndw, kernel_initializer=RN(seed=7), bias_initializer='zeros', name="CNNlayer3") )
        CNN.add( layers.BatchNormalization(name='BNlayer2') )
        CNN.add( layers.Activation('relu', name='reluCNNlayer3') )
        CNN.add( layers.Flatten(name="CNNflatened") )    

        DNN.add( layers.Dense(32, kernel_initializer=RN(seed=7), bias_initializer='zeros', name="DNNlayer1") )
        DNN.add( layers.Activation('relu', name='reluDNNlayer1') )
        DNN.add( layers.Dense(16, kernel_initializer=RN(seed=7), bias_initializer='zeros', name="DNNlayer2") )
        DNN.add( layers.Activation('relu', name='reluDNNlayer2') )
        DNN.add( layers.Dense(1, kernel_initializer=RN(seed=7), bias_initializer='zeros', name="DNNout") )
        DNN.add( layers.Activation('sigmoid', name='sigmoidDNNout') )

        CNNflatened = CNN(layers.Lambda(lambda x : x, name="CNNlayer0")(images))
        middleMan = layers.Concatenate(axis=1, name='middleMan')([CNNflatened, positions])
        TauIdentified = DNN(layers.Lambda(lambda x : x, name="TauIdentifier")(middleMan))

        TauIdentifierModel = keras.Model([images, positions], TauIdentified, name='TauCNNIdentifier')

        metrics2follow = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.AUC()]
        TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                   metrics=metrics2follow,
                                   run_eagerly=True)

        ############################## Model training ##############################

        history = TauIdentifierModel.fit([X1, X2], Y, epochs=30, batch_size=1024, verbose=1, validation_split=0.1)

        TauIdentifierModel.save(outdir + '/TauCNNIdentifier')

        for metric in history.history.keys():
            if 'val_' in metric: continue

            plt.plot(history.history[metric], label='Training dataset', lw=2)
            plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            if metric=='loss': plt.legend(loc='upper right')
            else:              plt.legend(loc='lower right')
            mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
            plt.savefig(outdir+'/TauCNNIdentifier_plots/'+metric+'.pdf')
            plt.close()

        w = TauIdentifierModel.layers[2].weights[0].numpy()
        h, b = np.histogram(w, bins=100)
        props = dict(boxstyle='square', facecolor='white', edgecolor='black')
        textstr1 = '\n'.join((r'% of zeros = {}'.format(np.sum(w==0)/np.size(w))))
        plt.figure(figsize=(10,10))
        plt.bar(b[:-1], h, width=b[1]-b[0])
        # plt.text(w.min()+0.01, h.max()-1, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        # plt.legend(loc = 'upper left')
        plt.grid(linestyle=':')
        plt.yscale('log')
        plt.xlabel('Weight value')
        plt.ylabel('Recurrence')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauCNNIdentifier_plots/CNN_sparsity.pdf')
        plt.close()

        w = TauIdentifierModel.layers[6].weights[0].numpy()
        h, b = np.histogram(w, bins=100)
        props = dict(boxstyle='square', facecolor='white', edgecolor='black')
        textstr1 = '\n'.join((r'% of zeros = {}'.format(np.sum(w==0)/np.size(w))))
        plt.figure(figsize=(10,10))
        plt.bar(b[:-1], h, width=b[1]-b[0])
        # plt.text(w.min()+0.01, h.max()-1, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        # plt.legend(loc = 'upper left')
        plt.grid(linestyle=':')
        plt.yscale('log')
        plt.xlabel('Weight value')
        plt.ylabel('Recurrence')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauCNNIdentifier_plots/DNN_sparsity.pdf')
        plt.close()

    else:
        TauIdentifierModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier', compile=False)

    ############################## Model validation ##############################

    X1_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Ident_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Ident_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/Y_Ident_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    train_ident = TauIdentifierModel.predict([X1, X2])
    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Y, train_ident)
    AUCtrain = metrics.roc_auc_score(Y, train_ident)

    valid_ident = TauIdentifierModel.predict([X1_valid, X2_valid])
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Y_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Y_valid, valid_ident)

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left')
    plt.xlim(0.8,1.01)
    # plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier_plots/validation_roc.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['true']  = Y_valid.ravel()
    plt.figure(figsize=(10,10))
    plt.hist(df[df['true']==1]['score'], bins=np.arange(0,1,0.05), label='Tau', color='green', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==0]['score'], bins=np.arange(0,1,0.05), label='PU', color='red', density=True, histtype='step', lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center')
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNIdentifier_plots/CNN_score.pdf')
    plt.close()


    ############################## Feature importance ##############################

    # since we have two inputs we pass a list of inputs to the explainer
#    explainer = shap.GradientExplainer(TauIdentifierModel, [X1, X2])
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
#    plt.savefig(outdir+'/TauCNNIdentifier_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNIdentifier_plots/shap1.pdf')
#    plt.close()

