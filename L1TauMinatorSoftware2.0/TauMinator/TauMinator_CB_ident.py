from tensorflow.keras.initializers import RandomNormal as RN
from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import cmsml
import sys
import os

np.random.seed(7)
tf.random.set_seed(7)

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def inspectWeights(model, which):
    if which=='kernel': idx=0
    if which=='bias':   idx=1

    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[idx].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure(figsize=(10,10))
    bins = np.linspace(-0.4, 0.4, 50)
    plt.hist(histosW,bins,histtype='step',stacked=True,label=labelsW)
    plt.legend(frameon=False,loc='upper left', fontsize=16)
    plt.ylabel('Recurrence')
    plt.xlabel('Weight value')
    plt.xlim(-0.7,0.5)
    plt.yscale('log')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/modelSparsity'+which+'.pdf')
    plt.close()

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                default=None)
    parser.add_option("--date",         dest="date",                             default=None)
    parser.add_option("--inTag",        dest="inTag",                            default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                        default="5x9")
    parser.add_option('--train',        dest='train',       action='store_true', default=False)
    parser.add_option('--dm_weighted',  dest='dm_weighted', action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training'+options.inTag
    if options.dm_weighted: outdir += '_dmWeighted' 
    os.system('mkdir -p '+outdir+'/TauMinator_CB_ident_plots')
    os.system('mkdir -p '+indir+'/CMSSWmodels')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(outdir+'/tensors/images_train.npz')['arr_0']
    X2 = np.load(outdir+'/tensors/posits_train.npz')['arr_0']
    Y  = np.load(outdir+'/tensors/target_train.npz')['arr_0']
    Yid  = Y[:,1].reshape(-1,1)

    if options.dm_weighted:
        # create dm weights
        bkg_sel = Yid.reshape(1,-1)[0] < 1
        Ydm = Y[:,4].reshape(-1,1)
        dm0 = Ydm.reshape(1,-1)[0] == 0
        dm1 = (Ydm.reshape(1,-1)[0] == 1) | (Ydm.reshape(1,-1)[0] == 2)
        dm10 = Ydm.reshape(1,-1)[0] == 10
        dm11 = (Ydm.reshape(1,-1)[0] == 11) | (Ydm.reshape(1,-1)[0] == 12)

        bkg_w = Yid.shape[0] / (Yid.shape[0] - (np.sum(dm0)+np.sum(dm1)+np.sum(dm10)+np.sum(dm11)))
        dm0_w = Yid.shape[0] / np.sum(dm0)
        dm1_w = Yid.shape[0] / np.sum(dm1)
        dm10_w = Yid.shape[0] / np.sum(dm10)
        dm11_w = Yid.shape[0] / np.sum(dm11)

        dm_weights = bkg_w*bkg_sel + dm0_w*dm0 + dm1_w*dm1 + dm10_w*dm10 + dm11_w*dm11


    ############################## Models definition ##############################

    # This model identifies the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauMinator_CB_ident_plots/training_ident.log')
        print(options)

        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        x = layers.Conv2D(16, (2,3), input_shape=(N, M, 3), use_bias=False, name="CNNlayer1")(images)
        x = layers.BatchNormalization(name='BN_CNNlayer1')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer1')(x)
        # x = layers.MaxPooling2D((2,2), name="MP_CNNlayer1")(x)
        x = layers.Conv2D(32, (2,3), use_bias=False, name="CNNlayer2")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer2')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer2')(x)
        # x = layers.MaxPooling2D((2,2), name="MP_CNNlayer2")(x)
        x = layers.Conv2D(32, (2,3), use_bias=False, name="CNNlayer3")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer3')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer3')(x)
        x = layers.Flatten(name="CNNflattened")(x)
        x = layers.Concatenate(axis=1, name='middleMan')([x, positions])
        
        y = layers.Dense(64, use_bias=False, name="IDlayer1")(x)
        y = layers.Activation('relu', name='RELU_IDlayer1')(y)
        y = layers.Dense(64, use_bias=False, name="IDlayer2")(y)
        y = layers.Activation('relu', name='RELU_IDlayer2')(y)
        y = layers.Dense(32, use_bias=False, name="IDlayer3")(y)
        y = layers.Activation('relu', name='RELU_IDlayer3')(y)
        y = layers.Dense(1, use_bias=False, name="IDout")(y)
        y = layers.Activation('sigmoid', name='sigmoid_IDout')(y)

        TauIdentified = y

        TauMinatorModel = keras.Model(inputs=[images, positions], outputs=TauIdentified, name='TauMinator_CB_indent')

        if options.dm_weighted:
            TauMinatorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                                    sample_weight_mode='sample-wise',
                                    run_eagerly=True)
        else:
            TauMinatorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
                                    run_eagerly=True)


        # print(TauMinatorModel.summary())
        # exit()

        ############################## Model training ##############################

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        if options.dm_weighted:
            history = TauMinatorModel.fit([X1, X2], Yid, epochs=200, batch_size=1024, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks, sample_weight=dm_weights)
        
        else:
            history = TauMinatorModel.fit([X1, X2], Yid, epochs=200, batch_size=1024, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks)

        TauMinatorModel.save(outdir + '/TauMinator_CB_ident')

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauMinator_CB_ident_plots/'+metric+'.pdf')
                plt.close()

            else:
                if 'val_' in metric: continue

                plt.plot(history.history[metric], label='Training dataset', lw=2)
                plt.plot(history.history['val_'+metric], label='Testing dataset', lw=2)
                plt.xlabel('Epoch')
                if 'loss' in metric:
                    plt.ylabel('Loss')
                    plt.legend(loc='upper right')
                elif 'auc' in metric:
                    plt.ylabel('AUC')
                    plt.legend(loc='lower right')
                elif 'binary_accuracy' in metric:
                    plt.ylabel('Binary accuracy')
                    plt.legend(loc='lower right')
                elif 'root_mean_squared_error' in metric:
                    plt.ylabel('Root Mean Squared Error')
                    plt.legend(loc='upper right')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauMinator_CB_ident_plots/'+metric+'.pdf')
                plt.close()

        ############################## Make split CNN and DNN models ##############################

        # save convolutional part alone
        image_in = TauMinatorModel.get_layer(index=0).get_output_at(0)
        posit_in = TauMinatorModel.get_layer(name='TowerClusterPosition').get_output_at(0)
        flat_out = TauMinatorModel.get_layer(name='middleMan').get_output_at(0)
        CNNmodel = tf.keras.Model([image_in, posit_in], flat_out, name="TauMinator_CB_conv")
        CNNmodel.save(outdir+'/CNNmodel', include_optimizer=False)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/CNNmodel_CB.pb', CNNmodel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/CNNmodel_CB.pb.txt', CNNmodel, variables_to_constants=True)

        # create middleman to be used for the saving of the id dnn alone
        idx = 0
        for layer in TauMinatorModel.layers:
            if layer._name == 'middleMan': idx += 1; break
            idx += 1
        input_shape = TauMinatorModel.layers[idx].get_input_shape_at(0)[1]
        middleMan = keras.Input(shape=input_shape, name='middleMan')
        
        # save id dense part alone
        x_dnn = middleMan
        for layer in TauMinatorModel.layers[idx:]:
            x_dnn = layer(x_dnn)
        ID_DNNmodel = tf.keras.Model(middleMan, x_dnn, name="TauMinator_CB_ident")
        ID_DNNmodel.save(outdir+'/ID_DNNmodel', include_optimizer=False)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/DNNident_CB.pb', ID_DNNmodel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/DNNident_CB.pb.txt', ID_DNNmodel, variables_to_constants=True)

        # validate the full model against the two split models
        X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
        X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
        Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
        Yid_valid  = Y_valid[:,1].reshape(-1,1)

        y_full = np.array( TauMinatorModel.predict([X1_valid, X2_valid]) )
        y_split  = np.array( ID_DNNmodel(CNNmodel([X1_valid, X2_valid])) )
        if not np.array_equal(y_full, y_split):
            print('\n\n****************************************************************')
            print(" WARNING : Full model and split ID model outputs do not match")
            print("           Output of np.allclose() = "+str(np.allclose(y_full, y_split)))
            print('****************************************************************\n\n')

        # restore normal output
        sys.stdout = sys.__stdout__

    else:
        TauMinatorModel = keras.models.load_model(outdir+'/TauMinator_CB_ident', compile=False)

        X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
        X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
        Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
        Yid_valid  = Y_valid[:,1].reshape(-1,1)

    ############################## Model validation and pots ##############################

    train_ident = TauMinatorModel.predict([X1, X2])
    valid_ident = TauMinatorModel.predict([X1_valid, X2_valid])

    FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(Yid, train_ident)
    AUCtrain = metrics.roc_auc_score(Yid, train_ident)

    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Yid_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Yid_valid, valid_ident)

    # save ID working points
    WP99 = np.interp(0.99, TPRvalid, THRvalid)
    WP95 = np.interp(0.95, TPRvalid, THRvalid)
    WP90 = np.interp(0.90, TPRvalid, THRvalid)
    WP85 = np.interp(0.85, TPRvalid, THRvalid)
    WP80 = np.interp(0.80, TPRvalid, THRvalid)
    WP75 = np.interp(0.75, TPRvalid, THRvalid)
    wp_dict = {
        'wp99' : WP99,
        'wp95' : WP95,
        'wp90' : WP90,
        'wp85' : WP85,
        'wp80' : WP80,
        'wp75' : WP75
    }
    save_obj(wp_dict, outdir+'/TauMinator_CB_ident_plots/CLTW_TauIdentifier_WPs.pkl')

    inspectWeights(TauMinatorModel, 'kernel')
    # inspectWeights(TauMinatorModel, 'bias')

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    # plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/validation_roc.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Training ROC, AUC = %.3f' % (AUCtrain),   color='blue',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.yscale('log')
    plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/validation_roc_log.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['true']  = Yid_valid.ravel()
    df['gen_pt'] = Y_valid[:,0].ravel()
    df['L1_et'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel()
    # df['gen_dm'] = Y_valid[:,4].ravel()
    
    plt.figure(figsize=(10,10))
    plt.hist(df[df['true']==1]['score'], bins=np.arange(0,1,0.05), label='Tau', color='green', density=True, histtype='step', lw=2)
    plt.hist(df[df['true']==0]['score'], bins=np.arange(0,1,0.05), label='PU', color='red', density=True, histtype='step', lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/CNN_score.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(df[df['true']==0]['score'], df[df['true']==0]['L1_et'], color='red', alpha=0.2)
    plt.scatter(df[df['true']==1]['score'], df[df['true']==1]['L1_et'], color='green', alpha=0.2)
    plt.grid(linestyle=':')
    # plt.xlim(0.5,1.001)
    plt.ylim(17,40)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel(r'CNN score')
    plt.ylabel(r'$p_{T}^{Gen. \tau}$')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/CNN_score_vs_l1et.pdf')
    plt.close()

    ################
    ## PER DM PLOTS
'''
    DMdict = {
                0  : r'$h^{\pm}$',
                1  : r'$h^{\pm}\pi^{0}$',
                10 : r'$h^{\pm}h^{\mp}h^{\pm}$',
                11 : r'$h^{\pm}h^{\mp}h^{\pm}\pi^{0}$',
            }


    bkg_sel_valid = Yid_valid.reshape(1,-1)[0] < 1
    Ydm_valid = Y_valid[:,4].reshape(-1,1)
    dm0_valid = Ydm_valid.reshape(1,-1)[0] == 0
    dm1_valid = (Ydm_valid.reshape(1,-1)[0] == 1) | (Ydm_valid.reshape(1,-1)[0] == 2)
    dm10_valid = Ydm_valid.reshape(1,-1)[0] == 10
    dm11_valid = (Ydm_valid.reshape(1,-1)[0] == 11) | (Ydm_valid.reshape(1,-1)[0] == 12)

    X1_valid_dm0 = X1_valid[dm0_valid | bkg_sel_valid]
    X2_valid_dm0 = X2_valid[dm0_valid | bkg_sel_valid]
    Yid_valid_dm0 = Yid_valid[dm0_valid | bkg_sel_valid]
    valid_ident_dm0 = TauMinatorModel.predict([X1_valid_dm0, X2_valid_dm0])
    FPRvalid_dm0, TPRvalid_dm0, THRvalid_dm0 = metrics.roc_curve(Yid_valid_dm0, valid_ident_dm0)
    AUCvalid_dm0 = metrics.roc_auc_score(Yid_valid_dm0, valid_ident_dm0)

    X1_valid_dm1 = X1_valid[dm1_valid | bkg_sel_valid]
    X2_valid_dm1 = X2_valid[dm1_valid | bkg_sel_valid]
    Yid_valid_dm1 = Yid_valid[dm1_valid | bkg_sel_valid]
    valid_ident_dm1 = TauMinatorModel.predict([X1_valid_dm1, X2_valid_dm1])
    FPRvalid_dm1, TPRvalid_dm1, THRvalid_dm1 = metrics.roc_curve(Yid_valid_dm1, valid_ident_dm1)
    AUCvalid_dm1 = metrics.roc_auc_score(Yid_valid_dm1, valid_ident_dm1)

    X1_valid_dm10 = X1_valid[dm10_valid | bkg_sel_valid]
    X2_valid_dm10 = X2_valid[dm10_valid | bkg_sel_valid]
    Yid_valid_dm10 = Yid_valid[dm10_valid | bkg_sel_valid]
    valid_ident_dm10 = TauMinatorModel.predict([X1_valid_dm10, X2_valid_dm10])
    FPRvalid_dm10, TPRvalid_dm10, THRvalid_dm10 = metrics.roc_curve(Yid_valid_dm10, valid_ident_dm10)
    AUCvalid_dm10 = metrics.roc_auc_score(Yid_valid_dm10, valid_ident_dm10)

    X1_valid_dm11 = X1_valid[dm11_valid | bkg_sel_valid]
    X2_valid_dm11 = X2_valid[dm11_valid | bkg_sel_valid]
    Yid_valid_dm11 = Yid_valid[dm11_valid | bkg_sel_valid]
    valid_ident_dm11 = TauMinatorModel.predict([X1_valid_dm11, X2_valid_dm11])
    FPRvalid_dm11, TPRvalid_dm11, THRvalid_dm11 = metrics.roc_curve(Yid_valid_dm11, valid_ident_dm11)
    AUCvalid_dm11 = metrics.roc_auc_score(Yid_valid_dm11, valid_ident_dm11)


    plt.figure(figsize=(10,10))
    plt.plot(TPRvalid_dm0, FPRvalid_dm0,   label=DMdict[0]+' ROC, AUC = %.3f' % (AUCvalid_dm0),   color='lime',lw=2)
    plt.plot(TPRvalid_dm1, FPRvalid_dm1,   label=DMdict[1]+' ROC, AUC = %.3f' % (AUCvalid_dm1),   color='blue',lw=2)
    plt.plot(TPRvalid_dm10, FPRvalid_dm10, label=DMdict[10]+' ROC, AUC = %.3f' % (AUCvalid_dm10), color='orange',lw=2)
    plt.plot(TPRvalid_dm11, FPRvalid_dm11, label=DMdict[11]+' ROC, AUC = %.3f' % (AUCvalid_dm11), color='fuchsia',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    # plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/validation_roc_perDM.pdf')
    plt.close()


    plt.figure(figsize=(10,10))
    plt.hist(df[df['gen_dm']==0]['score'],                       bins=np.arange(0,1,0.05), label=DMdict[0],  color='lime',    density=True, histtype='step', lw=2)
    plt.hist(df[(df['gen_dm']==1)|(df['gen_dm']==2)]['score'],   bins=np.arange(0,1,0.05), label=DMdict[1],  color='blue',    density=True, histtype='step', lw=2)
    plt.hist(df[df['gen_dm']==10]['score'],                      bins=np.arange(0,1,0.05), label=DMdict[10], color='orange',  density=True, histtype='step', lw=2)
    plt.hist(df[(df['gen_dm']==11)|(df['gen_dm']==12)]['score'], bins=np.arange(0,1,0.05), label=DMdict[11], color='fuchsia', density=True, histtype='step', lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper center', fontsize=16)
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/CNN_score_perDM.pdf')
    plt.close()


    plt.figure(figsize=(10,10))
    plt.plot(TPRvalid_dm0, FPRvalid_dm0,   label=DMdict[0]+' ROC, AUC = %.3f' % (AUCvalid_dm0),   color='lime',lw=2)
    plt.plot(TPRvalid_dm1, FPRvalid_dm1,   label=DMdict[1]+' ROC, AUC = %.3f' % (AUCvalid_dm1),   color='blue',lw=2)
    plt.plot(TPRvalid_dm10, FPRvalid_dm10, label=DMdict[10]+' ROC, AUC = %.3f' % (AUCvalid_dm10), color='orange',lw=2)
    plt.plot(TPRvalid_dm11, FPRvalid_dm11, label=DMdict[11]+' ROC, AUC = %.3f' % (AUCvalid_dm11), color='fuchsia',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.yscale('log')
    plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_ident_plots/validation_roc_perDM_log.pdf')
    plt.close()

'''