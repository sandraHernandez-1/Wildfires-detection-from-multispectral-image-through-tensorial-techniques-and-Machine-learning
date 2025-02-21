"""

Title          Class imbalance library
Authors       
               Eng. González Andrea
               Andrea.Gonzalez@cinvestav.mx
               
               Dr. López Josué
               Josue.Lopez@cinvestav.mx
               
               Dr. Torres Deni 
               Deni.Torres@cinvestav.mx
               
               Dr. Yañez Israel
               jyanez_ptc@upjr.edu.mx

               Centre for Advanced Research and Education of the National 
               Polytechnic Institute, CINVESTAV-IPN
               Guadalajara, México 
               http://gdl.cinvestav.mx

Date          November 14th 2020
Version       1.0
"""
#%%
import numpy as np
from sklearn.metrics import matthews_corrcoef,confusion_matrix

def confusion_matrix_to_vectors(Cm):
    classes = np.shape(Cm)[0]
    y_true = []
    y_pred = []
    for i in range (classes):
        for j in range (classes):
            m = Cm[i,j]
            if m != 0:
                for k in range (m):
                    y_true.append(i)
                    y_pred.append(j)
    return y_true, y_pred

def imbalance_parameters(mpp, mpn, mnp, mnn):
    mp = mpp + mpn
    mn = mnp + mnn
    m = mp + mn
    lambdaPP = mpp / mp
    lambdaNN = mnn / mn
    Delta = 2 *(mp/m) - 1  
    return lambdaPP, lambdaNN, Delta

def bin_CM(Cm,i):
    TP = Cm[i,i]
    FN = np.sum(Cm[i]) - Cm[i,i]
    FP = np.sum(Cm[:,i]) - Cm[i,i]
    TN = np.sum(Cm) - TP - FN - FP
    return TP,FN,FP,TN

def sns(Cm):
    C = np.shape(Cm)[0]
    sns_b = []
    sns = []
    sns_bias = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        sns.append(lambdaPP)
        sns_bias.append(0)
        sns_b.append(lambdaPP)
    return {'Balanced':sns_b,'Imbalanced':sns,'Bias':sns_bias}

def spc(Cm): 
    C = np.shape(Cm)[0]
    spc_b = []
    spc = []
    spc_bias = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        spc.append(lambdaNN) 
        spc_bias.append(0)
        spc_b.append(lambdaNN)
    return {'Balanced': spc_b,'Imbalanced':spc,'Bias':spc_bias}

def prc(Cm):
    C = np.shape(Cm)[0]
    prc = []
    prc_bias = []
    prc_b = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        prc.append((lambdaPP *(1 + Delta)) / (lambdaPP*(1 + Delta) + (1 - lambdaNN)*(1 - Delta)))
        prc_bias.append(((1 + Delta)/((1+Delta)+((1-lambdaNN)/(lambdaPP)) * (1 - Delta)))-(1/(1+((1-lambdaNN)/(lambdaPP)))))
        prc_b.append(lambdaPP / (lambdaPP + (1-lambdaNN)))
    return {'Balanced':prc_b,'Imbalanced':prc,'Bias':prc_bias}

def npv(Cm):
    C = np.shape(Cm)[0]
    npv = []
    npv_bias = []
    npv_b = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        npv.append((lambdaNN *(1 - Delta)) / (lambdaNN *(1 - Delta) + (1 - lambdaPP)*(1 + Delta)))
        npv_bias.append(((1 - Delta)/((1 - Delta)+((1-lambdaPP)/(lambdaNN)) * (1 + Delta)))-(1/(1+((1-lambdaPP)/(lambdaNN)))))
        npv_b.append(lambdaNN / (lambdaNN + (1-lambdaPP)))
    return {'Balanced':npv_b,'Imbalanced':npv,'Bias':npv_bias}

def acc(Cm):
    C = np.shape(Cm)[0]
    lambda_pp =[]
    lambda_nn = []
    deltas = []
    acc = []
    acc_bias = []
    acc_classic = []
    acc_b = []
    recall = 0
    recall_w = 0
    w = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        lambda_pp.append(lambdaPP)
        lambda_nn.append(lambdaNN)
        deltas.append(Delta)
        acc_classic.append((TP+TN)/(TP+FN+TN+FP))
        acc.append(lambdaPP *((1 + Delta) / 2) + lambdaNN *((1 - Delta) / 2))
        acc_bias.append((Delta/2)* (lambdaPP-lambdaNN))
        acc_b.append((lambdaPP + lambdaNN)/2)
        recall = recall + Cm[i,i] / np.sum(Cm[i,:])
        w.append(np.sum(Cm) / (C * np.sum(Cm[i,:])))
        recall_w = recall_w + (Cm[i,i] / np.sum(Cm[i,:])) * w[i]
    a_acc = np.mean(acc)
    a_acc_b = np.mean(acc_b)
    o_acc = np.trace(Cm)/(TP+FN+TN+FP)
    balanced_acc = recall/C
    w_balanced = recall_w / np.sum(w)
    return {'Classic':acc_classic,
            'Balanced':acc_b,
            'Imbalanced':acc,
            'Bias':acc_bias, 
            'Average_Acc': a_acc, 
            'Average_Acc_Balanced': a_acc_b,
            'Overall_Acc':o_acc,
            'Balanced_Acc': balanced_acc,
            'Balanced_Acc_weighted': w_balanced,
            'LambdaPP': lambda_pp,
            'LambdaNN': lambda_nn,
            'Delta': deltas}

def f1(Cm):
    C = np.shape(Cm)[0]
    f1 = []
    f1_bias = []
    f1_b = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        f1.append((2 * lambdaPP *(1 + Delta)) / (((1 + lambdaPP) * (1 + Delta)) + ((1 - lambdaNN)* (1 - Delta))))
        f1_bias.append(((2 * lambdaPP * (1+Delta))/((1+lambdaPP)*(1+Delta) + (1-lambdaNN)*(1-Delta)))-((2 * lambdaPP) / (2 + lambdaPP - lambdaNN)))
        f1_b.append(2*lambdaPP / (2+lambdaPP-lambdaNN))
    return {'Balanced':f1_b,'Imbalanced':f1,'Bias':f1_bias}

def gm(Cm):
    C = np.shape(Cm)[0]
    gm = []
    gm_bias = []
    gm_b = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        gm.append(np.sqrt(lambdaPP * lambdaNN))
        gm_bias.append(0)
        gm_b.append(np.sqrt(lambdaPP * lambdaNN))
    return {'Balanced':gm_b,'Imbalanced':gm,'Bias':gm_bias}

def mccn(Cm):
    C = np.shape(Cm)[0]
    mccn = []
    mccn_bias = []
    mccn_b = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        mccn.append(0.5 *((lambdaPP + lambdaNN - 1) / (np.sqrt((lambdaPP + (1-lambdaNN)*((1-Delta)/(1+Delta))) * (lambdaNN + (1-lambdaPP)* ((1+Delta)/(1-Delta)))))+1))
        mccn_bias.append(((lambdaPP+lambdaNN-1) / (2 * np.sqrt((lambdaPP+(1-lambdaNN)*((1-Delta)/(1+Delta)))*(lambdaNN+(1-lambdaPP)*((1+Delta)/(1-Delta)))))) - ((lambdaPP+lambdaNN-1) / (2* np.sqrt((lambdaPP+(1-lambdaNN))*(lambdaNN + (1-lambdaPP))))))
        mccn_b.append(1/2 *((lambdaPP + lambdaNN - 1) / (np.sqrt((lambdaPP + (1-lambdaNN)) * (lambdaNN + (1-lambdaPP))))+1))
    return {'Balanced':mccn_b,'Imbalanced':mccn,'Bias':mccn_bias}

def bmn(Cm):
    C = np.shape(Cm)[0]
    bmn = []
    bmn_bias = []
    bmn_b = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        bmn.append((lambdaPP + lambdaNN) / 2)
        bmn_bias.append(0)
        bmn_b.append((lambdaPP + lambdaNN) / 2)
    return {'Balanced':bmn_b,'Imbalanced':bmn,'Bias':bmn_bias}

def mkn(Cm):
    C = np.shape(Cm)[0]
    mkn = []
    mkn_bias = []
    mkn_b = []
    for i in range(C):
        TP,FN,FP,TN = bin_CM(Cm,i)
        lambdaPP, lambdaNN, Delta =  imbalance_parameters(TP,FN,FP,TN)
        mkn.append(0.5 * (((1+Delta)/((1+Delta)+((1-lambdaNN)/lambdaPP)*(1-Delta))) + ((1-Delta)/((1-Delta)+((1-lambdaPP)/lambdaNN)*(1+Delta)))))
        mkn_bias.append(0.5 * (((1+Delta)/((1+Delta)+((1-lambdaNN)/(lambdaPP))*(1-Delta))) - (1/(1+((1-lambdaNN)/lambdaPP))) + ((1-Delta)/((1-Delta)+((1-lambdaPP)/(lambdaNN))*(1+Delta))) - (1/(1+((1-lambdaPP)/(lambdaNN))))))
        mkn_b.append(0.5*((1/1+((1-lambdaNN)/(lambdaPP)))+(1/(1+((1-lambdaPP)/(lambdaNN))))))
    return {'Balanced':mkn_b,'Imbalanced':mkn,'Bias':mkn_bias}

def kp(Cm):
    C = np.shape(Cm)[0]
    ChanceAgree = 0
    sum_matrix = np.sum(Cm)
    Agree = np.trace(Cm) / sum_matrix
    for i in range(C):
        prob1 = np.sum(Cm[:,i]) / sum_matrix
        prob2 = np.sum(Cm[i,:]) / sum_matrix
        ChanceAgree_class = prob1 * prob2
        ChanceAgree = ChanceAgree + ChanceAgree_class
    kappa = (Agree-ChanceAgree)/(1-ChanceAgree)
    kappa = (kappa+1)/2
    return {'Kappa':kappa}


def report(y_test, y_pred):
    Cm = confusion_matrix(y_test,y_pred)
    classes = np.shape(Cm)[0]
    accs = acc(Cm)
    kappa = kp(Cm)
    mcc = (matthews_corrcoef(y_test, y_pred)+1)/2
    mccns = mccn(Cm)
    prcs = prc(Cm)
    npvs = npv(Cm)
    F1 = f1(Cm)
    Mkn= mkn(Cm)
    Gm = gm(Cm)
    metrics = [accs["Overall_Acc"],
               accs["Balanced_Acc"],
               accs["Balanced_Acc_weighted"],
               accs["Average_Acc"],
               kappa["Kappa"], mcc]
    print("OA = ",metrics[0])
    print("BA = ",metrics[1])
    print("BAW = ",metrics[2])
    print("AA = ",metrics[3])
    print("K = ",metrics[4])
    print("MCC = ",metrics[5])

    print("---------------------------------------------------------------------------------------------")
    print("Class",4*" ","PRC",7*" ","NPV",6*" ","ACC",6*" ","F1",6*" ","MCCn",6*" ","MKn",6*" ","GM")
    print("---------------------------------------------------------------------------------------------")
    for i in range(classes):
        print(i+1,7*" ","%.4f" % prcs["Imbalanced"][i],3*" ","%.4f" % npvs["Imbalanced"][i],3*" ",
              "%.4f" % accs["Imbalanced"][i],3*" ","%.4f" % F1["Imbalanced"][i],3*" ",
              "%.4f" % mccns["Imbalanced"][i],3*" ","%.4f" % Mkn["Imbalanced"][i],3*" ","%.4f" % np.mean(Gm["Imbalanced"][i]))
    print("Average",1*" ","%.4f" % np.mean(prcs["Imbalanced"]),3*" ","%.4f" % np.mean(npvs["Imbalanced"]),3*" ",
              "%.4f" % np.mean(accs["Imbalanced"]),3*" ","%.4f" % np.mean(F1["Imbalanced"]),3*" ",
              "%.4f" % np.mean(mccns["Imbalanced"]),3*" ","%.4f" % np.mean(Mkn["Imbalanced"]),3*" ","%.4f" % np.mean(Gm["Imbalanced"]))
    
    print("---------------------------------------------------------------------------------------------")
    print("Class",4*" ","PRC",7*" ","NPV",6*" ","ACC",6*" ","F1",6*" ","MCCn",6*" ","MKn",6*" ","GM")
    print("---------------------------------------------------------------------------------------------")
    for i in range(classes):
        print(i+1,7*" ","%.4f" % prcs["Bias"][i],3*" ","%.4f" % npvs["Bias"][i],3*" ",
              "%.4f" % accs["Bias"][i],3*" ","%.4f" % F1["Bias"][i],3*" ",
              "%.4f" % mccns["Bias"][i],3*" ","%.4f" % Mkn["Bias"][i],3*" ","%.4f" % Gm["Bias"][i])
    print("Average",1*" ","%.4f" % np.mean(prcs["Bias"]),3*" ","%.4f" % np.mean(npvs["Bias"]),3*" ",
              "%.4f" % np.mean(accs["Bias"]),3*" ","%.4f" % np.mean(F1["Bias"]),3*" ",
              "%.4f" % np.mean(mccns["Bias"]),3*" ","%.4f" % np.mean(Mkn["Bias"]),3*" ","%.4f" % np.mean(Gm["Bias"]))
    print("--------------------------------------------------------")
    print("Class",4*" ","LambdaPP",4*" ","LambdaNN",4*" ","Delta")
    print("--------------------------------------------------------")
    for i in range(classes):
        print(i+1,7*" ","%.4f" % accs["LambdaPP"][i],3*" ","%.4f" % accs["LambdaNN"][i],3*" ","%.4f" % accs["Delta"][i])
        
    return metrics




    
    
    


