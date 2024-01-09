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
    plt.savefig(outdir+'/TauMinator_CE_plots/modelSparsity'+which+'.pdf')
    plt.close()

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                      default="5x9")
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauMinator_CE_plots')
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
    X3 = np.load(outdir+'/tensors/shapes_train.npz')['arr_0']
    Y  = np.load(outdir+'/tensors/target_train.npz')['arr_0']
    Yid  = Y[:,1].reshape(-1,1)
    Ycal = Y[:,0].reshape(-1,1)

    ############################## Models definition ##############################

    # This model identifies the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    if options.train:
        # set output to go both to terminal and to file
        sys.stdout = Logger(outdir+'/TauMinator_CE_plots/training.log')
        print(options)

        images = keras.Input(shape = (N, M, 3), name='TowerClusterImage')
        positions = keras.Input(shape = 2, name='TowerClusterPosition')
        cl3dFeats = keras.Input(shape = 17, name='AssociatedCl3dFeatures')

        wndw = (2,2)
        if N <  5 and M >= 5: wndw = (1,2)
        if N <  5 and M <  5: wndw = (1,1)

        x = images
        x = layers.Conv2D(4, wndw, input_shape=(N, M, 3), use_bias=False, name="CNNlayer1")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer1')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer1')(x)
        x = layers.MaxPooling2D(wndw, name="MP_CNNlayer1")(x)
        x = layers.Conv2D(8, wndw, use_bias=False, name="CNNlayer2")(x)
        x = layers.BatchNormalization(name='BN_CNNlayer2')(x)
        x = layers.Activation('relu', name='RELU_CNNlayer2')(x)
        x = layers.Flatten(name="CNNflattened")(x)
        x = layers.Concatenate(axis=1, name='middleMan')([x, positions, cl3dFeats])
        
        y1 = layers.Dense(16, use_bias=False, name="IDlayer1")(x)
        y1 = layers.Activation('relu', name='RELU_IDlayer1')(y1)
        y1 = layers.Dense(8, use_bias=False, name="IDlayer2")(y1)
        y1 = layers.Activation('relu', name='RELU_IDlayer2')(y1)
        y1 = layers.Dense(1, use_bias=False, name="IDout")(y1)
        y1 = layers.Activation('sigmoid', name='sigmoid_IDout')(y1)
        
        y2 = layers.Dense(16, use_bias=False, name="CALlayer1")(x)
        y2 = layers.Activation('relu', name='RELU_CALlayer1')(y2)
        y2 = layers.Dense(8, use_bias=False, name="CALlayer2")(y2)
        y2 = layers.Activation('relu', name='RELU_CALlayer2')(y2)
        y2 = layers.Dense(1, use_bias=False, name="CALout")(y2)

        TauIdentified = y1
        TauCalibrated = y2

        TauMinatorModel = keras.Model(inputs=[images, positions, cl3dFeats], outputs=[TauIdentified, TauCalibrated], name='TauMinator_CE')

        def custom_MAPE(y_true, y_pred):
            # weighted_mape = tf.math.abs((y_true[1]-y_pred[1])/y_true[1]) * y_true[0]
            # return 2 * tf.math.reduce_sum(weighted_mape) / tf.math.reduce_sum(y_true[0])

            weighted_rmse = tf.math.sqrt( tf.math.square(y_true[1]-y_pred[1]) * y_true[0] / tf.math.reduce_sum(y_true[0]) ) 
            return weighted_rmse / 100

        metrics2follow = {'sigmoid_IDout' : [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                          # 'CALout'        : ['RootMeanSquaredError'] # this does not really make sense to track cause it will include also bkg and we do not care of that
                         }
        TauMinatorModel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True),
                                   loss=[tf.keras.losses.BinaryCrossentropy(from_logits=True), custom_MAPE],
                                   metrics=metrics2follow,
                                   run_eagerly=True)

        # print(TauMinatorModel.summary())
        # exit()

        ############################## Model training ##############################

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, mode='min', patience=10, verbose=1, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)]

        history = TauMinatorModel.fit([X1, X2, X3], [Yid, Ycal], epochs=200, batch_size=1024, shuffle=True, verbose=1, validation_split=0.25, callbacks=callbacks)

        TauMinatorModel.save(outdir + '/TauMinator_CE')

        for metric in history.history.keys():
            if metric == 'lr':
                plt.plot(history.history[metric], lw=2)
                plt.ylabel('Learning rate')
                plt.xlabel('Epoch')
                plt.yscale('log')
                mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
                plt.savefig(outdir+'/TauMinator_CE_plots/'+metric+'.pdf')
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
                plt.savefig(outdir+'/TauMinator_CE_plots/'+metric+'.pdf')
                plt.close()

        ############################## Make split CNN and DNN models ##############################

        # save convolutional part alone
        image_in = TauMinatorModel.get_layer(index=0).get_output_at(0)
        flat_out = TauMinatorModel.get_layer(name='middleMan').get_output_at(0)
        CNNmodel = tf.keras.Model([image_in, positions, cl3dFeats], flat_out)
        CNNmodel.save(outdir + '/CNNmodel', include_optimizer=False)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/CNNmodel_CE.pb', CNNmodel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/CNNmodel_CE.pb.txt', CNNmodel, variables_to_constants=True)

        # create middleman to be used for the saving of the id/cal dnns alone
        idx = 0
        for layer in TauMinatorModel.layers:
            if layer._name == 'middleMan': idx += 1; break
            idx += 1
        input_shape = TauMinatorModel.layers[idx].get_input_shape_at(0)[1]
        middleMan = keras.Input(shape=input_shape, name='middleMan')
        
        # save id dense part alone
        x_dnn = middleMan
        for layer in TauMinatorModel.layers[idx:]:
            if not 'ID' in layer.name: continue
            x_dnn = layer(x_dnn)
        ID_DNNmodel = tf.keras.Model(middleMan, x_dnn)
        ID_DNNmodel.save(outdir + '/ID_DNNmodel', include_optimizer=False)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/ID_DNNmodel_CE.pb', ID_DNNmodel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/ID_DNNmodel_CE.pb.txt', ID_DNNmodel, variables_to_constants=True)

        # save cal dense part alone
        x_dnn = middleMan
        for layer in TauMinatorModel.layers[idx:]:
            if not 'CAL' in layer.name: continue
            x_dnn = layer(x_dnn)
        CAL_DNNmodel = tf.keras.Model(middleMan, x_dnn)
        CAL_DNNmodel.save(outdir + '/CAL_DNNmodel', include_optimizer=False)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/CAL_DNNmodel_CE.pb', CAL_DNNmodel, variables_to_constants=True)
        cmsml.tensorflow.save_graph(indir+'/CMSSWmodels/CAL_DNNmodel_CE.pb.txt', CAL_DNNmodel, variables_to_constants=True)

        # validate the full model against the two split models
        X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
        X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
        X3_valid = np.load(outdir+'/tensors/shapes_valid.npz')['arr_0']
        Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
        Yid_valid  = Y_valid[:,1].reshape(-1,1)
        Ycal_valid = Y_valid[:,0].reshape(-1,1)

        y_full_id, y_full_cal = np.array( TauMinatorModel.predict([X1_valid, X2_valid]) )
        y_split_id  = np.array( ID_DNNmodel(CNNmodel([X1_valid, X2_valid])) )
        y_split_cal = np.array( CAL_DNNmodel(CNNmodel([X1_valid, X2_valid])) )
        if not np.array_equal(y_full_id, y_split_id):
            print('\n\n****************************************************************')
            print(" WARNING : Full model and split ID model outputs do not match")
            print("           Output of np.allclose() = "+str(np.allclose(y_full_id, y_split_id)))
            print('****************************************************************\n\n')

        if not np.array_equal(y_full_cal, y_split_cal):
            print('\n\n****************************************************************')
            print(" WARNING : Full model and split CAL model outputs do not match")
            print("           Output of np.allclose() = "+str(np.allclose(y_full_cal, y_split_cal)))
            print('****************************************************************\n\n')

        # restore normal output
        sys.stdout = sys.__stdout__

    else:
        TauMinatorModel = keras.models.load_model('/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training'+options.inTag+'/TauMinator_CE', compile=False)

        X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
        X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
        X3_valid = np.load(outdir+'/tensors/shapes_valid.npz')['arr_0']
        Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
        Yid_valid  = Y_valid[:,1].reshape(-1,1)
        Ycal_valid = Y_valid[:,0].reshape(-1,1)

    ############################## Model validation and pots ##############################

    train_ident, train_calib = TauMinatorModel.predict([X1, X2, X3])
    valid_ident, valid_calib = TauMinatorModel.predict([X1_valid, X2_valid, X3_valid])

    ####### ID PART #######

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
    save_obj(wp_dict, outdir+'/TauMinator_CE_plots/CLTW_TauIdentifier_WPs.pkl')

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
    plt.savefig(outdir+'/TauMinator_CE_plots/validation_roc.pdf')
    plt.close()

    df = pd.DataFrame()
    df['score'] = valid_ident.ravel()
    df['true']  = Yid_valid.ravel()
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
    plt.savefig(outdir+'/TauMinator_CE_plots/CNN_score.pdf')
    plt.close()


    ####### CALIB PART #######

    # dfLL = pd.DataFrame()
    # dfLL['calib_pt'] = train_calib.ravel()
    # dfLL['gen_pt']   = Ycal
    # dfLL['response'] = dfLL['calib_pt'] / dfLL['gen_pt']
    # # dfLL['gen_pt_binned']  = pd.cut(dfLL['gen_pt'],
    # #                                 bins=[18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500],
    # #                                 labels=False,
    # #                                 include_lowest=True)

    # dfLL['gen_pt_binned']  = ((dfLL['gen_pt'] - 18)/5).astype('int32')

    # meansTrainPt = dfLL.groupby('gen_pt_binned').mean()
    # meansTrainPt['logpt1'] = np.log(meansTrainPt['calib_pt'])
    # meansTrainPt['logpt2'] = meansTrainPt.logpt1**2
    # meansTrainPt['logpt3'] = meansTrainPt.logpt1**3
    # meansTrainPt['logpt4'] = meansTrainPt.logpt1**4

    # input_LL = meansTrainPt[['logpt1', 'logpt2', 'logpt3', 'logpt4']]
    # target_LL = meansTrainPt['response']
    # LogLinearizer = LinearRegression().fit(input_LL, target_LL)

    # select only the actual taus    
    tau_sel = Yid.reshape(1,-1)[0] > 0
    X1 = X1[tau_sel]
    Y = Y[tau_sel]
    train_calib = train_calib[tau_sel]

    # select only the actual taus
    tau_sel_valid = Yid_valid.reshape(1,-1)[0] > 0
    X1_valid = X1_valid[tau_sel_valid]
    Y_valid = Y_valid[tau_sel_valid]
    valid_calib = valid_calib[tau_sel_valid]

    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1).ravel()
    dfTrain['calib_pt']   = train_calib.ravel()
    dfTrain['gen_pt']     = Y[:,0].ravel()
    dfTrain['gen_eta']    = Y[:,2].ravel()
    dfTrain['gen_phi']    = Y[:,3].ravel()
    dfTrain['gen_dm']     = Y[:,4].ravel()
    
    # logpt1 = np.log(abs(dfTrain['calib_pt']))
    # logpt2 = logpt1**2
    # logpt3 = logpt1**3
    # logpt4 = logpt1**4
    # dfTrain['calib_pt_LL'] = dfTrain['calib_pt'] / LogLinearizer.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel()
    dfValid['calib_pt']   = valid_calib.ravel()
    dfValid['gen_pt']     = Y_valid[:,0].ravel()
    dfValid['gen_eta']    = Y_valid[:,2].ravel()
    dfValid['gen_phi']    = Y_valid[:,3].ravel()
    dfValid['gen_dm']     = Y_valid[:,4].ravel()

    # logpt1 = np.log(abs(dfValid['calib_pt']))
    # logpt2 = logpt1**2
    # logpt3 = logpt1**3
    # logpt4 = logpt1**4
    # dfValid['calib_pt_LL'] = dfValid['calib_pt'] / LogLinearizer.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/responses_comparison.pdf')
    plt.close()

    # PLOTS PER DM
    DMdict = {
            0  : r'$h^{\pm}$',
            1  : r'$h^{\pm}\pi^{0}$',
            10 : r'$h^{\pm}h^{\mp}h^{\pm}$',
            11 : r'$h^{\pm}h^{\mp}h^{\pm}\pi^{0}$',
        }

    tmp0 = dfValid[dfValid['gen_dm']==0]
    tmp1 = dfValid[(dfValid['gen_dm']==1) | (dfValid['gen_dm']==2)]
    tmp10 = dfValid[dfValid['gen_dm']==10]
    tmp11 = dfValid[(dfValid['gen_dm']==11) | (dfValid['gen_dm']==12)]
    plt.figure(figsize=(10,10))
    plt.hist(tmp0['uncalib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['uncalib_pt']/tmp0['gen_pt']), np.std(tmp0['uncalib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['uncalib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['uncalib_pt']/tmp1['gen_pt']), np.std(tmp1['uncalib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['uncalib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['uncalib_pt']/tmp10['gen_pt']), np.std(tmp10['uncalib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['uncalib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['uncalib_pt']/tmp11['gen_pt']), np.std(tmp11['uncalib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/uncalibrated_DM_responses_comparison.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(tmp0['calib_pt']/tmp0['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp0['calib_pt']/tmp0['gen_pt']), np.std(tmp0['calib_pt']/tmp0['gen_pt'])),      color='lime',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp1['calib_pt']/tmp1['gen_pt'],   bins=np.arange(0,5,0.1), label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp1['calib_pt']/tmp1['gen_pt']), np.std(tmp1['calib_pt']/tmp1['gen_pt'])),      color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp10['calib_pt']/tmp10['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp10['calib_pt']/tmp10['gen_pt']), np.std(tmp10['calib_pt']/tmp10['gen_pt'])), color='orange',lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(tmp11['calib_pt']/tmp11['gen_pt'], bins=np.arange(0,5,0.1), label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmp11['calib_pt']/tmp11['gen_pt']), np.std(tmp11['calib_pt']/tmp11['gen_pt'])), color='fuchsia',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/calibrated_DM_responses_comparison.pdf')
    plt.close()


    # 2D REPOSNSE VS ETA
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$|\eta^{Gen \tau}|$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/response_vs_eta_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PHI
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$\phi^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/response_vs_phi_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PT
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$p_{T}^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    # plt.xlim(-0.1,5)
    plt.xlim(0.0,2.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/response_vs_pt_comparison.pdf')
    plt.close()


    # L1 TO GEN MAPPING
    dfTrain['gen_pt_bin'] = pd.cut(dfTrain['gen_pt'],
                                   bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 2000],
                                   # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500, 1000, 2000],
                                   labels=False,
                                   include_lowest=True)

    dfValid['gen_pt_bin'] = pd.cut(dfValid['gen_pt'],
                                   bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 2000],
                                   # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500, 1000, 2000],
                                   labels=False,
                                   include_lowest=True)

    pt_bins_centers = np.arange(17.5,157.5,5)

    trainL1 = dfTrain.groupby('gen_pt_bin')['calib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['calib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['calib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['calib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/GenToCalibL1_pt.pdf')
    plt.close()

    trainL1 = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean()
    validL1 = dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean()
    trainL1std = dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std()
    validL1std = dfValid.groupby('gen_pt_bin')['uncalib_pt'].std()

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 uncalibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CE_plots/GenToUncalinL1_pt.pdf')
    plt.close()

    # PLOTS LOGLINEARIZER
    # plt.figure(figsize=(10,10))
    # plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfValid['calib_pt_LL']/dfValid['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated LogLinearized response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt_LL']/dfValid['gen_pt']), np.std(dfValid['calib_pt_LL']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    # plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    # plt.ylabel(r'a.u.')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/TauMinator_CE_plots/responses_comparison_valid_LL.pdf')
    # plt.close()

    # PLOTS LOGLINEARIZER
    # plt.figure(figsize=(10,10))
    # plt.hist(dfTrain['uncalib_pt']/dfTrain['gen_pt'], bins=np.arange(0,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['uncalib_pt']/dfTrain['gen_pt']), np.std(dfTrain['uncalib_pt']/dfTrain['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    # plt.hist(dfTrain['calib_pt_LL']/dfTrain['gen_pt'],   bins=np.arange(0,5,0.1), label=r'Valid. Calibrated LogLinearized response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt_LL']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt_LL']/dfTrain['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    # plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    # plt.ylabel(r'a.u.')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/TauMinator_CE_plots/responses_comparison_train_LL.pdf')
    # plt.close()

    # 2D REPOSNSE VS PT
    # plt.figure(figsize=(10,10))
    # plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated', alpha=0.2, color='red')
    # plt.scatter(dfValid['calib_pt_LL'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated LogLinearized', alpha=0.2, color='green')
    # plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    # plt.ylabel(r'$p_{T}^{Gen \tau}$')
    # plt.legend(loc = 'upper right', fontsize=16)
    # plt.grid(linestyle='dotted')
    # # plt.xlim(-0.1,5)
    # plt.xlim(0.0,2.0)
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/TauMinator_CE_plots/response_vs_pt_comparison_LL.pdf')
    # plt.close()
