#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:20:38 2023

Optuna をインストールしてね
https://aiacademy.jp/media/?p=2481


@author: setsuo.kinoshita
"""
# https://www.kaggle.com/code/kanncaa1/roc-curve-with-k-fold-cv/notebook
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
# https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/plot_roc_crossval.html

# Youden index
# https://pedimemo.com/python-youden-index/

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn import model_selection
from sklearn.calibration import CalibratedClassifierCV

#model
from sklearn.naive_bayes import GaussianNB
import optuna
import pickle
from sklearn.decomposition import PCA 
import knnimpute as knnimp
from scipy import stats
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.simplefilter('ignore')
##Set up Conditions
SEED = 2
##Select outlier proc
#outlier="HotDecK"
outlier="Remove"

##Select Classifier
ClassifierName="GNB"
#ClassifierName="SVC"
#ClassifierName="LogReg"

isLogitTransform=True

if __name__ == '__main__':
    cwd="E:/Dropbox/NipponTect/01_MRI/pMCI_onset_JADNI_MMSEZscore/0713/ML/"
    useCol=pd.read_excel(cwd+"useColName.xlsx").columns    
    df=pd.read_excel(cwd+'../Fix_HarmZ_JADNI_AD.xlsx')


    
    #--------------------------Remove outlier（ホットデック法）  https://qiita.com/ngayope330/items/7dce95abc42cfe6566bf
    df_stat=df.describe()
    df_stat.loc["outlierMAX",:]=df_stat.loc["75%",:]+(df_stat.loc["75%",:]-df_stat.loc["25%",:])*1.5
    df_stat.loc["outlierMIN",:]=df_stat.loc["25%",:]-(df_stat.loc["75%",:]-df_stat.loc["25%",:])*1.5
    ind_outlierMAX=df[useCol]>df_stat.loc["outlierMAX",:]
    ind_outlierMIN=df[useCol]<df_stat.loc["outlierMIN",:]
 
    
    df_tmp=df[useCol]
    df_tmp[(ind_outlierMAX|ind_outlierMIN)]=np.nan
    data=df_tmp.values
    ind_nan=np.isnan(data)

    #外れ値を含む被験者は除外
    if outlier=="Remove":
        df.dropna(axis=1)
    #ホットデック法で除外
    if outlier=="HotDeck":
        results=pd.DataFrame(knnimp.knn_impute_few_observed(data,ind_nan,3),columns=useCol)
        #results.to_excel("outlier_after.xlsx")
        #df[useCol].to_excel("outlier_befor.xlsx")
        df[useCol]=results    
    #---Extract Subdataset------------------------------------------------------------------------------------------------
    
    #indNLAD=((df.Diagnostic=="AD")|(df.Diagnostic=="NL")) #All visit
    indNLAD=((df.Diagnostic=="AD")|(df.Diagnostic=="NL"))&(df.VISCODE=="SC") #Only SC visit
    df_NLAD=df[indNLAD].reset_index(drop=True)
    df_psMCI=df[(df.subDiagnostic=="pMCI")|(df.subDiagnostic=="sMCI")].reset_index(drop=True)
    df_NLAD=df_NLAD.dropna(subset=useCol)
    df_psMCI=df_psMCI.dropna(subset=useCol)
    
    #-------T-test and draw each Zscore within ROI--------------------------------------------------------------------------------------------
    draw_Allplots=False #If false, draw only significant figure
    cols = ['NL','AD']
    plt.rcParams["font.size"] = 8
    df_ttest_results=pd.DataFrame()   
    sig_th=0.05/len(useCol) #ボンフェローニ
    cnt = 0
    for c in useCol:
        #ttest
        pval=stats.ttest_ind(df_NLAD.loc[df_NLAD.Diagnostic=="NL",c],df_NLAD.loc[df_NLAD.Diagnostic=="AD",c], equal_var=False)[1]
        if pval<sig_th:
            sig=True
        else:
            sig=False
        
        df_ttest_results=df_ttest_results.append(pd.DataFrame({"ROI":[c],"p_value":[pval],"Significant":[sig]}))
        
        if (draw_Allplots)|(sig):
            if cnt == 0:
                cnt = 1
                fig = plt.figure(figsize=(24, 16))
            axes = fig.add_subplot(4, 8, cnt)   #1行２列の１番目
            sns.violinplot(x='Diagnostic',y=c,data=df_NLAD,order=cols,ax=axes)
            txt="p="+str(pval)
            if sig:
                txt="* "+txt
            #axes.text(0.8, 0.95, txt, horizontalalignment='right', transform=axes.transAxes)
            axes.set_title(txt)
            cnt += 1
            if cnt > 4*8:
                plt.show()
                cnt = 0
    plt.show()
    
    sigCol=df_ttest_results[df_ttest_results.Significant].ROI
    #sigCol=["HarmZ-scoreRight Hippocampus",	"HarmZ-scoreLeft Hippocampus"]

    #############ハイパーパラメータの探索関数
    class Objective:
        def __init__(self, X, y,ClassifierName,SEED):
            self.X = X
            self.y = y
            self.ClassifierName=ClassifierName
            #乱数シードの固定
            #https://qiita.com/c60evaporator/items/633575c37863d1d18335
            self.study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=SEED))
        def __call__(self, trial):
            
            if self.ClassifierName=="GNB":
                params = {'var_smoothing': trial.suggest_uniform('var_smoothing', 1e-14, 1e-1),}
                clf = GaussianNB()
                clf.set_params(**params)
                score = model_selection.cross_val_score(clf,X=self.X, y=self.y, scoring='roc_auc', cv=10)
                return score.mean()
            if self.ClassifierName=="SVC":
                params = {
                    'C': trial.suggest_loguniform('C', 1e-5, 1),
                    'gamma': trial.suggest_loguniform('gamma', 1e-5, 1 ),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid'])
                }
                clf = SVC()
                clf.set_params(**params)
                
                score = model_selection.cross_val_score(clf,X=self.X, y=self.y, scoring='roc_auc', cv=10)
                return score.mean()

            if self.ClassifierName=="LogReg":
                params = {
                    # 最適化アルゴリズムを指定
                    'solver' : trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                    # 正則化の強さを指定（0.0001から10まで）
                    'C': trial.suggest_loguniform('C', 0.0001, 10),
                    # 最大反復回数（＊ソルバーが収束するまで）
                    'max_iter': trial.suggest_int('max_iter', 100, 100000)
                    }

                clf = LogisticRegression()
                clf.set_params(**params)
                
                score = model_selection.cross_val_score(clf,X=self.X, y=self.y, scoring='roc_auc', cv=10)

                
                return score.mean()
        def optimize(self,n_trials):
            self.study.optimize(self,n_trials=n_trials)
            if self.ClassifierName=="GNB":
                clf = GaussianNB(**self.study.best_params)
                self.model = CalibratedClassifierCV(clf, method='isotonic', cv=5)
                self.model.fit(self.X, self.y)
                
            if self.ClassifierName=="SVC":
                clf = SVC(**self.study.best_params,probability=True)
                self.model = CalibratedClassifierCV(clf, method='isotonic', cv=5)
                self.model.fit(self.X, self.y)

                #calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
                #calibrated_clf.fit(X_train, y_train)
                #calibratedpred[test_index] = calibrated_clf.predict_proba(X_valid)[:,1]
            if ClassifierName=="LogReg":
                self.model= LogisticRegression(**self.study.best_params)
                self.model = CalibratedClassifierCV(clf, method='isotonic', cv=5)
                self.model.fit(self.X, self.y)

                
    #### 入力データの準備    
    #X = df_NLAD.loc[:, useCol]
    X = df_NLAD.loc[:, sigCol]
    y = df_NLAD.Diagnostic=="AD"

    #X_test=df_psMCI.loc[:, useCol]
    X_test=df_psMCI.loc[:, sigCol]
    y_test=df_psMCI.subDiagnostic=="pMCI"                  
    
    #-------PCA--------------------------------------------------------------------------------------------
    numPCA=0
    if numPCA!=0:
        print("exe. PCA ")
        pca=PCA()
        pca.fit(X)
        X=pca.transform(X)[:,1:numPCA]
        X_test=pca.transform(X_test)[:,1:numPCA]
        

    ###Learning   
    #############ハイパーパラメータの探索

    print("Classifier:"+ClassifierName)
    objective=Objective(X, y, ClassifierName,SEED)
    objective.optimize(n_trials=100)

    
    ## Print Results of Traiging Dataset
    print("Number of finished trials: {}".format(len(objective.study.trials)))
    print("Best trial:")
    trial = objective.study.best_trial
    
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    #学習
    objective.model.fit(X, y)
    
    #作った学習機を保存する場合
    #pickeled_f = './GaussianNB_model.sav'
    #pickle.dump(model, open(pickeled_f, 'wb'))
    
    #読み込む場合
    #model = pickle.load(open(pickeled_f, 'rb'))
    
    #Probability(確率値?)
   # if ClassifierName=="LogReg":
   #     df_NLAD['Prob'] = objective.model.predict_log_proba(X)[:,1]    
   # else:
       #df_NLAD['Prob'] = objective.model.predict_proba(X)[:,1]
    df_NLAD['Prob'] = objective.model.predict_proba(X)[:,1]
   
    #判別結果を出す
    df_NLAD['Judge'] =objective.model.predict(X)

   # if ClassifierName=="LogReg":
   #     df_psMCI["Prob"]=objective.model.predict_log_proba(X_test)[:,1]
   # else :
        #df_psMCI["Prob"]=objective.model.predict_proba(X_test)[:,1]
    df_psMCI["Prob"]=objective.model.predict_proba(X_test)[:,1]
    df_psMCI["Predict"]=objective.model.predict(X_test)

    
    
    if numPCA==0:
        df_psMCI.to_excel(cwd+ClassifierName+"_noPCA_spMCI_Fix_HarmZ_JADNI_AD.xlsx",index=False)
        df_NLAD.to_excel(cwd+ClassifierName+"_noPCA_NLAD_Fix_HarmZ_JADNI_AD.xlsx",index=False)
        
    else:
        df_psMCI.to_excel(cwd+ClassifierName+"_PCA"+str(numPCA)+"_spMCI_Fix_HarmZ_JADNI_AD.xlsx",index=False)
        df_NLAD.to_excel(cwd+ClassifierName+"_PCA"+str(numPCA)+"_NLAD_Fix_HarmZ_JADNI_AD.xlsx",index=False)
    
    
    ##############Draw #####################
    #ボックスプロット、バイオリンプロット、経験的累積分布関数
    for graphtype in ['Box','Violin','Ecdf']:
       plt.rcParams["font.size"] = 20
       fig, axes = plt.subplots(1, 1, figsize=(12, 8))
       cols = ['NL','AD']
       if graphtype == 'Box':
           sns.boxplot(x='Diagnostic',y='Prob',data=df_NLAD,order=cols,ax=axes)
       else:
           if graphtype == 'Violin':
               sns.violinplot(x='Diagnostic',y='Prob',data=df_NLAD,order=cols,ax=axes)
               sns.swarmplot(y="Prob", x="Diagnostic", data=df_NLAD,order=cols, color="white",ax=axes)
           else:
               sns.ecdfplot(x='Prob',data=df_NLAD,hue="Diagnostic",ax=axes)
       axes.set_title('Demo Title')
       plt.show()
       
       
    ##############Draw ROC Curve#####################
    #Calc Roc
    fpr, tpr, thresholds = roc_curve(y,df_NLAD['Prob'])
    roc_auc = auc(fpr, tpr)
    
    test_fpr, test_tpr, test_thresholds = roc_curve(y_test,objective.model.predict(X_test))
    test_roc_auc = auc(test_fpr, test_tpr)
    
    Youden_index_candidates = tpr-fpr
    y_index = np.where(Youden_index_candidates==max(Youden_index_candidates))[0][0]
    cutoff = thresholds[y_index]

    sensitivity = tpr[y_index]
    specificity = 1 - fpr[y_index]
  
    y_pred = np.where(df_NLAD['Prob'] >= cutoff, 1, 0)
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y_true=y,y_pred=y_pred)
    
    #Draw Graph
    plt.rcParams["font.size"] = 26
    # plt.rcParams['font.family'] = 'Hiragino Sans' # for max
    plt.rcParams['font.family'] = 'MS Gothic' # for windows
    #　使えそうなフォントは以下のコマンドで調べてね
    # import matplotlib.font_manager
    # print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    
    fig = plt.figure(1, figsize=(12,12))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr, color='blue', lw=4, alpha=0.5, label='LOOCV ROC (AUC = %0.4f)' % (roc_auc))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
 
    total_num = len(y)
    MCI_num = int(y.sum())
    NL_num = total_num - MCI_num
    
    txt = 'n=' + str(total_num) + ' (NL:' + str(NL_num) + ' AD:' + str(MCI_num)+ ')'
    ax.text(0.99, 0.25, txt, horizontalalignment='right', transform=ax.transAxes)
    ax.text(0.99, 0.2, 'Youden_index (cutoff=' + str(round(cutoff,6))+')', horizontalalignment='right', transform=ax.transAxes)
    txt = 'sensitivity = ' + str(round(sensitivity,4))
    ax.text(0.99, 0.15, txt, horizontalalignment='right', transform=ax.transAxes)
    txt = 'specificity = ' + str(round(specificity,4))
    ax.text(0.99, 0.1, txt, horizontalalignment='right', transform=ax.transAxes)
    
    plt.title('GNB_ROC')
    plt.legend(loc="lower right")
    plt.show() 
    
    
    
       

