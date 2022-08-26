from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow.keras.initializers import RandomNormal as RN
from tensorflow.keras import layers, models
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
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

    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
    TauIdentifierModel = prune.prune_low_magnitude(TauIdentifierModel, **pruning_params)

    # metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.AUC()]
    TauIdentifierModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], run_eagerly=True)


    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifierPruning'+options.caloClNxM+'Training'+options.inTag

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

    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifierPruning'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauCNNIdentifierPruning_plots')

    history = TauIdentifierModel.fit([X1, X2], Y, epochs=10, batch_size=128, verbose=1, validation_split=0.1)

    TauIdentifierModel = strip_pruning(TauIdentifierModel)
    TauIdentifierModel.save(outdir + '/TauCNNIdentifierPruning')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/loss.pdf')
    plt.close()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/accuracy.pdf')
    plt.close()

    # plt.plot(history.history['FP'])
    # plt.plot(history.history['val_FP'])
    # plt.title('False Positives')
    # plt.ylabel('False Positive Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/false_positives.pdf')
    # plt.close()

    # plt.plot(history.history['TP'])
    # plt.plot(history.history['val_TP'])
    # plt.title('True Positives')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/true_positives.pdf')
    # plt.close()

    # plt.plot(history.history['FN'])
    # plt.plot(history.history['val_FN'])
    # plt.title('False Negatives')
    # plt.ylabel('False Negative Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/false_negatives.pdf')
    # plt.close()

    # plt.plot(history.history['TN'])
    # plt.plot(history.history['val_TN'])
    # plt.title('True Negatives')
    # plt.ylabel('True Negative Rate')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/true_negatives.pdf')
    # plt.close()

    w = TauIdentifierModel.layers[0].weights[0].numpy()
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
    plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/model_sparsity.pdf')
    plt.close()

    ############################## Model validation ##############################

    X1_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNValidator'+options.caloClNxM+'/X_Ident_CNN_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    X2_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNValidator'+options.caloClNxM+'/X_Ident_Dense_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    Y_valid  = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNValidator'+options.caloClNxM+'/Y_Ident_'+options.caloClNxM+'_forValidator.npz')['arr_0']

    train_ident = TauIdentifierModel.predict([X1, X2])
    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Y, train_ident)

    valid_ident = TauIdentifierModel.predict([X1_valid, X2_valid])
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Y_valid, valid_ident)

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC',   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC', color='green',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left')
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/validation_roc.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['true']  = Y_valid.ravel()
    plt.figure(figsize=(10,10))
    plt.hist(df[df['true']==1]['score'], bins=np.arange(-14,1,0.33), label='Tau', color='green', density=True, alpha=0.5)
    plt.hist(df[df['true']==0]['score'], bins=np.arange(-14,1,0.33), label='PU', color='red', density=True, alpha=0.5)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left')
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/CNN_score.pdf')
    plt.close()


    ############################## Feature importance ##############################

#    # since we have two inputs we pass a list of inputs to the explainer
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
#    plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/shap0.pdf')
#    plt.close()
#
#    plt.figure(figsize=(10,10))
#    # here we plot the explanations for all classes for the second input (this is the conv-net input)
#    shap.image_plot([shap_values[i][1] for i in range(len(shap_values))], X2_valid[:3], show=False)
#    plt.savefig(outdir+'/TauCNNIdentifierPruning_plots/shap1.pdf')
#    plt.close()

