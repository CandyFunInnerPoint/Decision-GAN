import json
import numpy as np
import pandas as pd
import datetime as dt
import copy
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
import scipy
import scipy.stats
import os

from itertools import * 

from datetime import date

from collections import OrderedDict
import ot

import math

class Generator(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim, mu_dim):
        super(Generator, self).__init__()
        self.linear1 = torch.nn.Linear(z_dim + x_dim, h_dim)
        self.linear2 = torch.nn.Linear(h_dim, 1)
        self.graph = torch.nn.Sequential(
	    self.linear1,
	    torch.nn.ReLU(),
	    self.linear2,
	)
        
    def forward(self, x):
        retEstList = []
        for featVecSymbol in x:
            retEst = self.graph(featVecSymbol)
            retEstList.append(retEst)
        retVecEst = torch.cat(retEstList, 1)
        return retVecEst

class Discriminator(nn.Module):
    def __init__(self, D_input_dim, x_dim, h_dim):
        super(Discriminator, self).__init__()
        self.linear1 = torch.nn.Linear(D_input_dim + x_dim, 2 * h_dim)
        self.linear2 = torch.nn.Linear(2 * h_dim, 1)
        
    def forward(self, x):
        self.graph = torch.nn.Sequential(
	    self.linear1,
	    torch.nn.ReLU(),
	    self.linear2
	)
        return self.graph(x)


class DAT_GAN(object):
    def __init__(self, true_mu, true_cov, train_df, eval_df_list, true_beta, z_dim, h_dim, isStacking, loss_weight_dict, warm_up_samples, train_num_samples, eval_num_samples, batch_size, strat_train_days, strat_val_days, n_epoch, n_critics, n_actors, lr, clip_lb, clip_ub, shrink, isPrintOpen, evalFreq, isUsingNewData, train_risk_gamma, eval_risk_gamma, dateToday, subFolderID, isRealData, isParaGen, sourceEtfNameList, halfLifeList, estHalfLife, targetEtfNameList, trainHorizonRange, evelHorizonRange, horizonDecay, hasAutoRegBias, hasFeatureInLoss):
        self.GPU = torch.cuda.is_available()
        print(self.GPU)
        #part1: true world
        self.true_mu = true_mu
        self.true_cov = true_cov
        self.train_df = train_df
        self.eval_df_list = eval_df_list
        assert(self.train_df in self.eval_df_list)
        self.true_beta = true_beta
        if not isRealData:
            assert(train_num_samples == eval_num_samples)#u have assumption here
        inv_cov = np.linalg.inv(self.true_cov) 
        numerator = np.matmul(inv_cov, np.ones((self.true_cov.shape[0], 1)))
        denominator = np.matmul(np.ones((1, self.true_cov.shape[0])), numerator)
        self.true_portWeights = numerator / denominator
        self.true_performance_mean = np.dot(self.true_portWeights.flatten(), self.true_mu)
        self.true_performance_var = np.matmul(np.matmul(self.true_portWeights.transpose(), self.true_cov), self.true_portWeights)

        #part3: training parameters
        self.warm_up_samples = warm_up_samples
        self.train_num_samples = train_num_samples #256
        if (not isRealData) or (isRealData and isParaGen):
            self.eval_num_samples = eval_num_samples #256 
        self.batch_size = batch_size #16#v in the paper
        self.strat_train_days = strat_train_days #32#n in the paper
        self.strat_val_days = strat_val_days #32#m in the paper
        self.n_epoch = n_epoch #20000
        self.n_critics = n_critics #1
        self.n_actors = n_actors #1
        self.lr = lr #1e-6
        self.clip_lb = clip_lb # -0.01
        self.clip_ub = clip_ub # 0.01
        self.shrink = shrink
        self.isPrintOpen = isPrintOpen
        self.evalFreq = evalFreq
        self.isUsingNewData = isUsingNewData
        self.train_risk_gamma = train_risk_gamma
        self.eval_risk_gamma_list = eval_risk_gamma_list
        assert(self.train_risk_gamma in self.eval_risk_gamma_list)
        self.dateToday = dateToday
        self.subFolderID = subFolderID

        self.isRealData = isRealData
        self.isParaGen = isParaGen
        self.sourceEtfNameList = sourceEtfNameList
        self.halfLifeList = halfLifeList
        self.estHalfLife = estHalfLife
        self.targetEtfNameList = targetEtfNameList
        self.numSourceEtf = len(self.sourceEtfNameList)
        self.numTargetEtf = len(self.targetEtfNameList)
        assert(self.sourceEtfNameList == self.targetEtfNameList)
        self.trainHorizonRange = trainHorizonRange
        self.evalHorizonRange = evalHorizonRange
        self.horizonDecay = horizonDecay
        self.hasAutoRegBias = hasAutoRegBias
        self.hasFeatureInLoss = hasFeatureInLoss

        #part2: discriminator, loss functions and generator models
        self.z_dim = z_dim #128
        self.h_dim = h_dim #16
        self.mu_dim = self.true_cov.shape[0]
        assert(len(self.sourceEtfNameList) == self.mu_dim)
        self.cov_dim = self.mu_dim ** 2

        self.D_input_dim_eval_dict = self.init_D_input_dim_eval_dict(self.mu_dim, self.cov_dim)
    
        self.isStacking = isStacking
        self.loss_weight_dict = OrderedDict(loss_weight_dict)
        print(self.loss_weight_dict)
        
        

             


        if self.isRealData:
            #real data
            print("realdata")
            self.dataForML, self.x_dim = self.genMLInputData()
            self.train_data, self.eval_data, self.eval_num_samples = self.sampleRealData(self.train_num_samples)
            self.train_data = torch.from_numpy(self.train_data)
            self.eval_data = torch.from_numpy(self.eval_data)
        else:
            #simulated data
            self.train_data = self.Sampling(self.true_mu.tolist(), self.true_cov, self.train_df, self.train_num_samples, self.strat_train_days, self.strat_val_days, "train")
            self.eval_data_dict = OrderedDict()
            for df in self.eval_df_list:
                self.eval_data_dict[df] = torch.from_numpy(self.Sampling(self.true_mu.tolist(), self.true_cov, df, self.eval_num_samples, self.strat_train_days, self.strat_val_days, "eval"))

        #part4: file related paths
        self.fileNamePostfix = "tDf" + str(self.train_df) + "zD" + str(self.z_dim) + "hD" + str(self.h_dim) + "muD" + str(self.mu_dim) + "iS" + str(self.isStacking) + "sll" + "".join(self.loss_weight_dict.keys()) + "isNSam" + str(self.train_num_samples) + "evalNSam" + str(self.eval_num_samples) + "bS" + str(self.batch_size) + "stTDays" + str(self.strat_train_days) + "stVDays" + str(self.strat_val_days) + "nEp" + str(self.n_epoch) + "nCr" + str(self.n_critics) + "lr" + str(self.lr) + "cLb" + str(self.clip_lb) + "cUb" + str(self.clip_ub) + "shrink" + str(self.shrink) + "tU" + str(self.train_risk_gamma) + "eU" + "".join(map(str, self.eval_risk_gamma_list)) + "iR" + str(self.isRealData) + "iP" + str(self.isParaGen) + "xD" + str(self.x_dim) + "tH" + str(self.trainHorizonRange) + "eH" + str(self.evalHorizonRange) + "hD" + str(self.horizonDecay) + "hA" + str(self.hasAutoRegBias) + "hF" + str(self.hasFeatureInLoss)
                
        #Part: write trueWorld stats file
        self.output_trueWorld_file = open('out/' + self.dateToday + '/' + str(self.subFolderID) + '/trueWorld_' + self.fileNamePostfix + ".csv", 'w', newline = '')
        csv.writer(self.output_trueWorld_file).writerow(["true_performance_mean", "true_performance_var"] + ["true_ret_mean"] * self.mu_dim + ["true_ret_cov"] * self.cov_dim + ["true_portWeights_mean"] * self.mu_dim + ["true_portWeights_cov"] * self.cov_dim)
        csv.writer(self.output_trueWorld_file).writerow(self.true_performance_mean.flatten().tolist() + self.true_performance_var.flatten().tolist() + self.true_mu.flatten().tolist() + self.true_cov.flatten().tolist() + self.true_portWeights.flatten().tolist() + [0] * self.cov_dim)
        self.output_trueWorld_file.close() 

        #discriminator
        if self.isRealData:
            if self.isStacking:
                self.D_input_dim_train_dict, self.discriminator_dict, self.discriminator_solver_dict = self.init_discriminator_train_dict_isStacking_RT(self.loss_weight_dict, self.mu_dim, self.cov_dim, self.x_dim, self.h_dim)
            else:
                self.D_input_dim_train_dict, self.discriminator_dict, self.discriminator_solver_dict = self.init_discriminator_train_dict(self.loss_weight_dict, self.mu_dim, self.cov_dim, self.x_dim, self.h_dim)
        else:
            if self.isStacking:
                self.D_input_dim_train_dict, self.discriminator, self.discriminator_solver = self.init_discriminator_train_dict(self.loss_weight_dict, self.mu_dim, self.cov_dim, self.x_dim, self.h_dim)
            else:
                self.D_input_dim_train_dict, self.discriminator_dict, self.discriminator_solver_dict = self.init_discriminator_train_dict(self.loss_weight_dict, self.mu_dim, self.cov_dim, self.x_dim, self.h_dim)


        #part: generator
        self.generator = Generator(self.z_dim, int(self.x_dim/len(self.targetEtfNameList)), self.h_dim, self.mu_dim).cuda()
        self.generator_solver = optim.Adam(self.generator.parameters(), lr=self.lr)

        self.output_generator_file = open('out/' + self.dateToday + '/' + str(self.subFolderID) + '/generator_' + self.fileNamePostfix + ".pt", 'wb')    

        print("generator's state_dict:")
        for param_tensor in self.generator.state_dict():
            print(param_tensor, "\t", self.generator.state_dict()[param_tensor].size())

        #Part7: write eval stats file
        self.output_trainWorld_file = open('out/' + self.dateToday + '/' + str(self.subFolderID) + '/trainWorld_' + self.fileNamePostfix + ".csv", 'w', newline = '')
        if self.isRealData:
            csv.writer(self.output_trainWorld_file).writerow(self.computeStatsName_RT("train", self.D_input_dim_eval_dict))
        else:
            csv.writer(self.output_trainWorld_file).writerow(self.computeStatsName("train", [self.train_df], [self.train_risk_gamma], self.D_input_dim_eval_dict))
        
        if self.isRealData:
            self.train_D_input_dict_RT = self.computeDInputHistDataFromHist(self.train_data, self.D_input_dim_eval_dict, self.evalHorizonRange)
            csv.writer(self.output_trainWorld_file).writerow(self.computeStats_RT(self.train_D_input_dict_RT))
        else:
            train_data_dict = OrderedDict()
            train_data_dict[self.train_df] = torch.from_numpy(self.train_data)
            self.train_D_input_dict = self.computeDInput(train_data_dict, [self.train_risk_gamma], self.D_input_dim_eval_dict, "eval")[0]
            csv.writer(self.output_trainWorld_file).writerow(self.computeStats([self.train_D_input_dict]))
        self.output_trainWorld_file.close()   


        self.output_evalWorld_file = open('out/' + self.dateToday + '/' + str(self.subFolderID) + '/evalWorld_' + self.fileNamePostfix + ".csv", 'w', newline = '')
        if self.isRealData:
            csv.writer(self.output_evalWorld_file).writerow(self.computeStatsName_RT("eval", self.D_input_dim_eval_dict))
        else:
            csv.writer(self.output_evalWorld_file).writerow(self.computeStatsName("eval", self.eval_df_list, self.eval_risk_gamma_list, self.D_input_dim_eval_dict))
        
#        print("eval_D_input_dict: ", eval_D_input_dict)
        if self.isRealData:
            self.eval_D_input_dict_RT = self.computeDInputHistDataFromHist(self.eval_data, self.D_input_dim_eval_dict, self.evalHorizonRange)
            csv.writer(self.output_evalWorld_file).writerow(self.computeStats_RT(self.eval_D_input_dict_RT))
        else:
            self.eval_D_input_dict_list = self.computeDInput(self.eval_data_dict, self.eval_risk_gamma_list, self.D_input_dim_eval_dict, "eval")
            csv.writer(self.output_evalWorld_file).writerow(self.computeStats(self.eval_D_input_dict_list))
        self.output_evalWorld_file.close()
        
        #part: training loss file
        self.output_fakeWorld_file = open('out/' + self.dateToday + '/' + str(self.subFolderID) + '/fakeWorld_' + self.fileNamePostfix + ".csv", 'w', newline = '')
        if self.isRealData:
            csv.writer(self.output_fakeWorld_file).writerow(["epoch", "D_loss", "G_loss"] + self.computeDGLossName_RT() + #self.computeStatsName_RT("gan", self.D_input_dim_eval_dict) + 
self.computeDistName_RT("wass", self.D_input_dim_eval_dict, "train") + self.computeDistName_RT("wass", self.D_input_dim_eval_dict, "eval"))
        else:
            if self.isStacking:
                csv.writer(self.output_fakeWorld_file).writerow(["epoch", "D_loss", "G_loss"] + self.computeStatsName("gan", [self.train_df], self.eval_risk_gamma_list, self.D_input_dim_eval_dict) + self.computeDistName("wass", [self.train_df], [self.train_risk_gamma], self.D_input_dim_eval_dict, "train") + self.computeDistName("wass", self.eval_df_list, self.eval_risk_gamma_list, self.D_input_dim_eval_dict, "eval"))
            else:
                csv.writer(self.output_fakeWorld_file).writerow(["epoch", "D_loss", "G_loss"] + self.computeDGLossName() + self.computeStatsName("gan", [self.train_df], self.eval_risk_gamma_list, self.D_input_dim_eval_dict) + self.computeDistName("wass", [self.train_df], [self.train_risk_gamma], self.D_input_dim_eval_dict, "train") + self.computeDistName("wass", self.eval_df_list, self.eval_risk_gamma_list, self.D_input_dim_eval_dict, "eval")) 

    def init_D_input_dim_eval_dict(self, mu_dim, cov_dim):
        D_input_dim_dict = OrderedDict()
        D_input_dim_dict["ret"] = mu_dim
        D_input_dim_dict["retEst"] = mu_dim
        D_input_dim_dict["covEst"] = cov_dim
        D_input_dim_dict["invCovEst"] = cov_dim
        D_input_dim_dict["portW"] = mu_dim
        D_input_dim_dict["per"] = 1
        D_input_dim_dict["utility"] = 1
        return D_input_dim_dict

    def init_discriminator_train_dict_isStacking_RT(self, loss_weight_dict, mu_dim, cov_dim, x_dim, h_dim):
        D_input_dim_dict = OrderedDict()
        total_dim_ret_per_utility = 0
        total_dim_retEst_covEst_invCovEst_portW = 0
        discriminator_dict = OrderedDict()
        discriminator_solver_dict = OrderedDict()
        for lossName, weight in loss_weight_dict.items():
            if lossName == "ret":
                D_input_dim_dict["ret"] = mu_dim
                total_dim_ret_per_utility += mu_dim
            elif lossName == "retEst":
                D_input_dim_dict["retEst"] = mu_dim
                total_dim_retEst_covEst_invCovEst_portW += mu_dim
            elif lossName == "covEst":
                D_input_dim_dict["covEst"] = cov_dim
                total_dim_retEst_covEst_invCovEst_portW += cov_dim
            elif lossName == "invCovEst":
                D_input_dim_dict["invCovEst"] = cov_dim
                total_dim_retEst_covEst_invCovEst_portW += cov_dim
            elif lossName == "portW":
                D_input_dim_dict["portW"] = mu_dim
                total_dim_retEst_covEst_invCovEst_portW += mu_dim
            elif lossName == "per":
                D_input_dim_dict["per"] = 1
                total_dim_ret_per_utility += 1
            elif lossName == "utility":
                D_input_dim_dict["utility"] = 1
                total_dim_ret_per_utility += 1
        if total_dim_ret_per_utility > 0:
            discriminator_dict["ret_per_utility"] = Discriminator(total_dim_ret_per_utility, x_dim, h_dim).cuda()
            discriminator_solver_dict["ret_per_utility"] = optim.Adam(discriminator_dict["ret_per_utility"].parameters(), lr = lr)
            self.loss_weight_dict["ret_per_utility"] = 1
        if total_dim_retEst_covEst_invCovEst_portW > 0:
            discriminator_dict["retEst_covEst_invCovEst_portW"] = Discriminator(total_dim_retEst_covEst_invCovEst_portW, x_dim, h_dim).cuda()
            discriminator_solver_dict["retEst_covEst_invCovEst_portW"] = optim.Adam(discriminator_dict["retEst_covEst_invCovEst_portW"].parameters(), lr = lr)
            self.loss_weight_dict["retEst_covEst_invCovEst_portW"] = 1
        return D_input_dim_dict, discriminator_dict, discriminator_solver_dict

    def init_discriminator_train_dict(self, loss_weight_dict, mu_dim, cov_dim, x_dim, h_dim):
        D_input_dim_dict = OrderedDict()
        if self.isStacking:
            total_dim = 0
        else:
            discriminator_dict = OrderedDict()
            discriminator_solver_dict = OrderedDict()
        for lossName, weight in loss_weight_dict.items():
            if lossName == "ret":
                D_input_dim_dict["ret"] = mu_dim
                if self.isStacking:
                    total_dim += mu_dim * (self.strat_train_days + self.strat_val_days)
                else:
                    discriminator_dict["ret"] = Discriminator(mu_dim, x_dim, h_dim).cuda()
                    discriminator_solver_dict["ret"] = optim.Adam(discriminator_dict["ret"].parameters(), lr=lr)
            elif lossName == "retEst":
                D_input_dim_dict["retEst"] = mu_dim
                if self.isStacking:
                    total_dim += mu_dim
                else:
                    discriminator_dict["retEst"] = Discriminator(mu_dim, x_dim, h_dim).cuda()
                    discriminator_solver_dict["retEst"] = optim.Adam(discriminator_dict["retEst"].parameters(), lr=lr)
            elif lossName == "covEst":
                D_input_dim_dict["covEst"] = cov_dim
                if self.isStacking:
                    total_dim += cov_dim
                else:
                    discriminator_dict["covEst"] = Discriminator(cov_dim, x_dim, h_dim).cuda()
                    discriminator_solver_dict["covEst"] = optim.Adam(discriminator_dict["covEst"].parameters(), lr=lr)
            elif lossName == "invCovEst":
                D_input_dim_dict["invCovEst"] = cov_dim
                if self.isStacking:
                    total_dim += cov_dim
                else:
                    discriminator_dict["invCovEst"] = Discriminator(cov_dim, x_dim, h_dim).cuda()
                    discriminator_solver_dict["invCovEst"] = optim.Adam(discriminator_dict["invCovEst"].parameters(), lr=lr)
            elif lossName == "portW":
                D_input_dim_dict["portW"] = mu_dim
                if self.isStacking:
                    total_dim += mu_dim
                else:
                    discriminator_dict["portW"] = Discriminator(mu_dim, x_dim, h_dim).cuda()
                    discriminator_solver_dict["portW"] = optim.Adam(discriminator_dict["portW"].parameters(), lr=lr)
            elif lossName == "per":
                D_input_dim_dict["per"] = 1
                if self.isStacking:
                    total_dim += 1 * self.strat_val_days
                else:
                    discriminator_dict["per"] = Discriminator(1, x_dim, h_dim).cuda() 
                    discriminator_solver_dict["per"] = optim.Adam(discriminator_dict["per"].parameters(), lr=lr)
            elif lossName == "utility":
                D_input_dim_dict["utility"] = 1
                if self.isStacking:
                    total_dim += 1 * self.strat_val_days
                else:
                    discriminator_dict["utility"] = Discriminator(1, x_dim, h_dim).cuda()
                    discriminator_solver_dict["utility"] = optim.Adam(discriminator_dict["utility"].parameters(), lr=lr)
        if self.isStacking:
            discriminator = Discriminator(total_dim, x_dim, h_dim).cuda()
            discriminator_solver = optim.Adam(discriminator.parameters(), lr = lr)
            return D_input_dim_dict, discriminator, discriminator_solver
        else:
            return D_input_dim_dict, discriminator_dict, discriminator_solver_dict

    def genMLInputData(self):
        #step 0: singler ticker features from singler hist data
        mergedData = pd.DataFrame()

        if self.isParaGen:
            np_data = self.SequentialSampling(self.true_mu.tolist(), self.true_cov, self.train_df, self.true_beta, self.warm_up_samples, self.train_num_samples + self.eval_num_samples)              
            mergedData = pd.DataFrame(data = np_data, columns = [etfName + ".DailyRet" for etfName in self.targetEtfNameList])
            for etfName in self.targetEtfNameList:
                for halfLife in self.halfLifeList:
                    mergedData[etfName + ".DailyRetEMA" + str(halfLife)] = mergedData[etfName + ".DailyRet"].ewm(halflife = halfLife, adjust = False, ignore_na = True).mean()
            print(mergedData.head())
        else:
            count = 0
            for etfName in self.targetEtfNameList:
                count += 1
                data = pd.read_csv("sectorETF/" + etfName + ".csv", skiprows = 1, header = 0)
       
                #featDailyRet
                data[etfName + ".DailyRet"] = (data["Close"] - data["Close"].shift(periods = 1)) * 1.0 / data["Close"].shift(periods = 1)

                #featDailyRetEMA
                for halfLife in self.halfLifeList:
                    data[etfName + ".DailyRetEMA" + str(halfLife)] = data[etfName + ".DailyRet"].ewm(halflife = halfLife, adjust = False, ignore_na = True).mean()

                data = data.drop(columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"])
                if count == 1:
                    mergedData = data
                else:
                    mergedData = mergedData.merge(data, left_on = "Date", right_on = "Date")

 
        dataForML = pd.DataFrame()
        dataForMLColInd = 0

        #feat
        self.featColIndBEListDailyRetSymbolDict = OrderedDict()
        self.featColIndBEListDailyRetEMADictSymbolDict = OrderedDict()
        for etfName in self.targetEtfNameList:
            #featDailyRet
            self.featColIndBEListDailyRetSymbolDict[etfName] = []
            self.featColIndBEListDailyRetSymbolDict[etfName].append(dataForMLColInd)
            dataForML[etfName + ".DailyRet"] = mergedData[etfName + ".DailyRet"]
            dataForMLColInd += 1
            self.featColIndBEListDailyRetSymbolDict[etfName].append(dataForMLColInd)
            #featDailyRetEMA
            self.featColIndBEListDailyRetEMADictSymbolDict[etfName] = OrderedDict()
            for halfLife in self.halfLifeList:
                self.featColIndBEListDailyRetEMADictSymbolDict[etfName][halfLife] = []
                self.featColIndBEListDailyRetEMADictSymbolDict[etfName][halfLife].append(dataForMLColInd)
                dataForML[etfName + ".DailyRetEMA" + str(halfLife)] = mergedData[etfName + ".DailyRetEMA" + str(halfLife)]
                dataForMLColInd += 1
                self.featColIndBEListDailyRetEMADictSymbolDict[etfName][halfLife].append(dataForMLColInd)


        x_dim = dataForMLColInd
        
        #warmUpTgtDailyRetEst
        self.warmUpTgtDailyRetEstColIndBEList = []
        self.warmUpTgtDailyRetEstColIndBEList.append(dataForMLColInd)
        for etfName in self.targetEtfNameList:
            dataForML[etfName + ".DailyRetEstCurrent"] = dataForML[etfName + ".DailyRet"].ewm(halflife = self.estHalfLife, adjust = False, ignore_na = True).mean()
        dataForMLColInd += self.numSourceEtf
        self.warmUpTgtDailyRetEstColIndBEList.append(dataForMLColInd)

        #warmUpTgtDailyCovEst
        self.warmUpTgtDailyCovEstColIndBEList = []
        self.warmUpTgtDailyCovEstColIndBEList.append(dataForMLColInd)
        for etfName1 in self.targetEtfNameList:
            for etfName2 in self.targetEtfNameList:
                dailyRetMult = dataForML[etfName1 + ".DailyRet"] * dataForML[etfName2 + ".DailyRet"]
                dataForML[etfName1 + "." + etfName2 + ".DailyCovEstCurrent"] = dailyRetMult.ewm(halflife = self.estHalfLife, adjust = False, ignore_na = True).mean()
        dataForMLColInd += self.numSourceEtf ** 2
        self.warmUpTgtDailyCovEstColIndBEList.append(dataForMLColInd)

        sample_size = dataForML.shape[0]
        one = np.ones((sample_size, self.numSourceEtf))
        diagMat = np.eye(self.numSourceEtf)

        #warmUpTgtDailyPortW
        self.warmUpTgtDailyPortWColIndBEListDict = OrderedDict()
        dailyCov = dataForML[dataForML.columns[self.warmUpTgtDailyCovEstColIndBEList[0]:self.warmUpTgtDailyCovEstColIndBEList[1]]].to_numpy().reshape((-1, self.numSourceEtf, self.numSourceEtf))
        dailyInvCov = Variable(torch.from_numpy(dailyCov * (1 - self.shrink)  + diagMat * self.shrink).float().cuda()).inverse().cpu().numpy()
        for risk_gamma in self.eval_risk_gamma_list:
            warmUpTgtDailyPortWColIndBEList = []
            warmUpTgtDailyPortWColIndBEList.append(dataForMLColInd)
            dailyPortW = self.compPortW(dataForML[dataForML.columns[self.warmUpTgtDailyRetEstColIndBEList[0]:self.warmUpTgtDailyRetEstColIndBEList[1]]].to_numpy(), dailyInvCov, one, risk_gamma)
            portInd = 0
            for etfName in self.targetEtfNameList:
                dataForML[etfName + ".DailyPortWCurrent" + str(risk_gamma)] = dailyPortW[:,portInd]
                portInd += 1
            dataForMLColInd += self.numSourceEtf
            warmUpTgtDailyPortWColIndBEList.append(dataForMLColInd)
            self.warmUpTgtDailyPortWColIndBEListDict[risk_gamma] = warmUpTgtDailyPortWColIndBEList

        estDecay = math.exp(math.log(0.5) / self.estHalfLife)
        one = np.ones((sample_size, self.numTargetEtf))
        diagMat = np.repeat(np.eye(self.numTargetEtf).reshape(1, self.numTargetEtf, self.numTargetEtf), sample_size, axis = 0)

        self.tgtDailyRetColIndBEListList = []

        self.tgtDailyPerColIndBEListDictList = []
        self.tgtDailyUtilityColIndBEListDictList = []

        self.tgtDailyRetEstColIndBEListList = []
        self.tgtDailyCovEstColIndBEListList = []
        self.tgtDailyInvCovEstColIndBEListList = []
        self.tgtDailyPortWColIndBEListDictList = []
        
        for horizon in range(self.evalHorizonRange):
            #part1: fundamental varaible of current day 

            #tgtDailyRet
            tgtDailyRetColIndBEList = []
            tgtDailyRetColIndBEList.append(dataForMLColInd)
            for etfName in self.targetEtfNameList:
                dataForML[etfName + ".DailyRetForward" + str(horizon)] = dataForML[etfName + ".DailyRet"].shift(periods = -horizon - 1)
            dataForMLColInd += self.numTargetEtf
            tgtDailyRetColIndBEList.append(dataForMLColInd)
            self.tgtDailyRetColIndBEListList.append(tgtDailyRetColIndBEList)

            
            #part2: per/utility evaluation of prev day
            #tgtDailyPer
            tgtDailyPerColIndBEListDict = OrderedDict()
            for risk_gamma in self.eval_risk_gamma_list:
                tgtDailyPerColIndBEList = []
                tgtDailyPerColIndBEList.append(dataForMLColInd)
                if horizon == 0: 
                    dataForML["DailyPerFoward" + str(horizon) + "rg" + str(risk_gamma)] = np.squeeze(np.matmul(np.expand_dims(dataForML[dataForML.columns[self.warmUpTgtDailyPortWColIndBEListDict[risk_gamma][0]:self.warmUpTgtDailyPortWColIndBEListDict[risk_gamma][1]]].to_numpy(), axis = 1), np.expand_dims(dataForML[dataForML.columns[self.tgtDailyRetColIndBEListList[horizon][0]: self.tgtDailyRetColIndBEListList[horizon][1]]].to_numpy(), axis = 2)), axis = (1, 2))
                else:
                    dataForML["DailyPerFoward" + str(horizon) + "rg" + str(risk_gamma)] = np.squeeze(np.matmul(np.expand_dims(dataForML[dataForML.columns[self.tgtDailyPortWColIndBEListDictList[horizon - 1][risk_gamma][0]: self.tgtDailyPortWColIndBEListDictList[horizon - 1][risk_gamma][1]]].to_numpy(), axis = 1), np.expand_dims(dataForML[dataForML.columns[self.tgtDailyRetColIndBEListList[horizon][0]: self.tgtDailyRetColIndBEListList[horizon][1]]].to_numpy(), axis = 2)), axis = (1, 2))
                dataForMLColInd += 1
                tgtDailyPerColIndBEList.append(dataForMLColInd)
                tgtDailyPerColIndBEListDict[risk_gamma] = tgtDailyPerColIndBEList
            self.tgtDailyPerColIndBEListDictList.append(tgtDailyPerColIndBEListDict)
 
           #tgtDailyUtility
            tgtDailyUtilityColIndBEListDict = OrderedDict()
            for risk_gamma in self.eval_risk_gamma_list:
                tgtDailyUtilityColIndBEList = []
                tgtDailyUtilityColIndBEList.append(dataForMLColInd)
                dailyPer = dataForML[dataForML.columns[self.tgtDailyPerColIndBEListDictList[horizon][risk_gamma][0]: self.tgtDailyPerColIndBEListDictList[horizon][risk_gamma][1]]].to_numpy()
                dataForML["DailyUtilityFoward" + str(horizon) + "rg" + str(risk_gamma)] =  dailyPer - risk_gamma * dailyPer ** 2
                dataForMLColInd += 1
                tgtDailyUtilityColIndBEList.append(dataForMLColInd)
                tgtDailyUtilityColIndBEListDict[risk_gamma] = tgtDailyUtilityColIndBEList
            self.tgtDailyUtilityColIndBEListDictList.append(tgtDailyUtilityColIndBEListDict)


            #part3: estimators of next day 
            if horizon < self.evalHorizonRange - 1:
                #tgtDailyRetEst
                tgtDailyRetEstColIndBEList = []
                tgtDailyRetEstColIndBEList.append(dataForMLColInd)
                if horizon == 0:
                    for etfName in self.targetEtfNameList:
                        dataForML[etfName + ".DailyRetEstForward" + str(horizon)] = dataForML[etfName + ".DailyRetEstCurrent"] * estDecay + dataForML[etfName + ".DailyRetForward" + str(horizon)] * (1 - estDecay)
                else:
                    for etfName in self.targetEtfNameList:
                        dataForML[etfName + ".DailyRetEstForward" + str(horizon)] = dataForML[etfName + ".DailyRetEstForward" + str(horizon - 1)] * estDecay + dataForML[etfName + ".DailyRetForward" + str(horizon)] * (1 - estDecay)
                dataForMLColInd += self.numTargetEtf
                tgtDailyRetEstColIndBEList.append(dataForMLColInd)
                self.tgtDailyRetEstColIndBEListList.append(tgtDailyRetEstColIndBEList)

                #tgtDailyCovEst
                tgtDailyCovEstColIndBEList = []
                tgtDailyCovEstColIndBEList.append(dataForMLColInd)
                if horizon == 0:
                    for etfName1 in self.targetEtfNameList:
                        for etfName2 in self.targetEtfNameList:
                            dataForML[etfName1 + "." + etfName2 + ".DailyCovEstForward" + str(horizon)] = dataForML[etfName1 + "." + etfName2 + ".DailyCovEstCurrent"] * estDecay + dataForML[etfName1 + ".DailyRetForward" + str(horizon)] * dataForML[etfName2 + ".DailyRetForward" + str(horizon)] * (1 - estDecay)
                else:
                    for etfName1 in self.targetEtfNameList:
                        for etfName2 in self.targetEtfNameList:
                            dataForML[etfName1 + "." + etfName2 + ".DailyCovEstForward" + str(horizon)] = dataForML[etfName1 + "." + etfName2 + ".DailyCovEstForward" + str(horizon - 1)] * estDecay + dataForML[etfName1 + ".DailyRetForward" + str(horizon)] * dataForML[etfName2 + ".DailyRetForward" + str(horizon)] * (1 - estDecay)
                dataForMLColInd += self.numTargetEtf ** 2
                tgtDailyCovEstColIndBEList.append(dataForMLColInd)
                self.tgtDailyCovEstColIndBEListList.append(tgtDailyCovEstColIndBEList)

                #tgtDailyInvCovEst
                tgtDailyInvCovEstColIndBEList = []
                tgtDailyInvCovEstColIndBEList.append(dataForMLColInd)
                tempMat = dataForML[dataForML.columns[self.tgtDailyCovEstColIndBEListList[horizon][0]:self.tgtDailyCovEstColIndBEListList[horizon][1]]].to_numpy().reshape((-1, self.numTargetEtf, self.numTargetEtf))
                dailyInvCov = Variable(torch.from_numpy(tempMat * (1 - self.shrink) + diagMat * self.shrink).float().cuda()).inverse().cpu()
                covMatRowInd = 0
                covMatColInd = 0
                for etfName1 in self.targetEtfNameList:
                    for etfName2 in self.targetEtfNameList:
                        dataForML[etfName1 + "." + etfName2 + ".DailyInvCovEstForward" + str(horizon)] = dailyInvCov[:, covMatRowInd, covMatColInd]
                        covMatColInd += 1
                    covMatRowInd += 1
                    covMatColInd = 0
                dataForMLColInd += self.numTargetEtf ** 2
                tgtDailyInvCovEstColIndBEList.append(dataForMLColInd)
                self.tgtDailyInvCovEstColIndBEListList.append(tgtDailyInvCovEstColIndBEList)

                #tgtDailyPortW
                tgtDailyPortWColIndBEListDict = OrderedDict()
                for risk_gamma in self.eval_risk_gamma_list:
                    tgtDailyPortWColIndBEList = []
                    tgtDailyPortWColIndBEList.append(dataForMLColInd)
                    dailyPortW = self.compPortW(dataForML[dataForML.columns[self.tgtDailyRetEstColIndBEListList[horizon][0]:self.tgtDailyRetEstColIndBEListList[horizon][1]]].to_numpy(), dataForML[dataForML.columns[self.tgtDailyInvCovEstColIndBEListList[horizon][0]:self.tgtDailyInvCovEstColIndBEListList[horizon][1]]].to_numpy().reshape((-1, self.numTargetEtf, self.numTargetEtf)), one, risk_gamma)
                    portWInd = 0
                    for etfName in self.targetEtfNameList:
                        dataForML[etfName + ".DailyPortWForward" + str(horizon) + "rg" + str(risk_gamma)] = dailyPortW[:, portWInd]
                        portWInd += 1
                    dataForMLColInd += self.numTargetEtf
                    tgtDailyPortWColIndBEList.append(dataForMLColInd)
                    tgtDailyPortWColIndBEListDict[risk_gamma] = tgtDailyPortWColIndBEList
                self.tgtDailyPortWColIndBEListDictList.append(tgtDailyPortWColIndBEListDict)


        pd.set_option("display.max_columns", None)

        dataForML.dropna(inplace = True)

        dataForML.to_csv("out/MLInputData_feat" + str(x_dim) + ".csv", index = False)

        return dataForML, x_dim

    def compPortWTorch(self, retEst, invCovEst, one, risk_gamma):
        retEst = torch.unsqueeze(retEst, 2)
        one = torch.unsqueeze(one, 2)
        one_invCovEst = torch.matmul(one.transpose(1,2), invCovEst)
        one_invCovEst_one = torch.matmul(one_invCovEst, one)
        one_invCovEst_retEst = torch.matmul(one_invCovEst, retEst)
        lamb = torch.div(one_invCovEst_retEst - 2 * risk_gamma, one_invCovEst_one)
        portW = torch.squeeze(torch.matmul(invCovEst, torch.div(retEst - lamb * one, 2 * risk_gamma)), 2)
        return portW

    def compPortW(self, retEst, invCovEst, one, risk_gamma):
        retEst = np.expand_dims(retEst, axis = 2)
        one = np.expand_dims(one, axis = 2)
        one_invCovEst = np.matmul(np.transpose(one, (0, 2, 1)), invCovEst)
        one_invCovEst_one = np.matmul(one_invCovEst, one)
        one_invCovEst_retEst = np.matmul(one_invCovEst, retEst)
        lamb = np.divide(one_invCovEst_retEst - 2 * risk_gamma, one_invCovEst_one)
        portW = np.squeeze(np.matmul(invCovEst, np.divide(retEst - np.multiply(np.repeat(lamb, one.shape[1], axis = 1), one), 2 * risk_gamma)), axis = 2)
        
        return portW

    def sampleRealData(self, train_num_samples):
        data = self.dataForML.to_numpy()
        eval_num_samples = data.shape[0] - train_num_samples
        train_data = data[:train_num_samples]
        eval_data = data[train_num_samples:]
        return train_data, eval_data, eval_num_samples

    def Sampling(self, mu, cov, df, num_samples, strat_train_days, strat_val_days, train_eval):
        if self.isUsingNewData:
            data = multivariate_t(mu, cov, df, num_samples * (strat_train_days + strat_val_days))
            data = data.reshape((num_samples, strat_train_days + strat_val_days, len(mu)))
            np.save(train_eval + "_sim_data_" + str(df) + ".npy", data)
        else:
            data = np.load(train_eval + "_sim_data_" + str(df) + ".npy")

        return data

    def SequentialSampling(self, mu, cov, df, beta, warm_up_samples, num_samples):
        if self.isUsingNewData:
            beta = np.array(beta).reshape(-1, 1)
            eps = multivariate_t(mu, cov, df, warm_up_samples + num_samples)
            data = np.zeros((num_samples, self.mu_dim))
            dailyRet = np.random.multivariate_normal(mu, cov, 1).T
            featList = [dailyRet]

            dailyRetEMADict = OrderedDict()
            decayDict = OrderedDict()
            for halfLife in self.halfLifeList:
                dailyRetEMADict[halfLife] = dailyRet
                featList.append(dailyRetEMADict[halfLife])
                decayDict[halfLife] = math.exp(math.log(0.5) / halfLife)
            featArr = np.concatenate(featList, axis = 1)
            for i in range(1, warm_up_samples + num_samples):
                dailyRet = np.matmul(featArr, beta) + eps[i - warm_up_samples, :].reshape(-1, 1)
                if i > warm_up_samples:
                    data[i - warm_up_samples, :] = dailyRet.T
                featList =[dailyRet]
                for halfLife in self.halfLifeList:
                    dailyRetEMADict[halfLife] = dailyRetEMADict[halfLife] * decayDict[halfLife] + dailyRet * (1 - decayDict[halfLife])
                    featList.append(dailyRetEMADict[halfLife])
                featArr = np.concatenate(featList, axis = 1)

            np.save("seq_sample_sim_data_" + str(df) + ".npy", data)
        else:
            data = np.load("seq_sample_sim_data_" + str(df) + ".npy")
        return data
            
        
    def computeDInput(self, df_data_dict, risk_gamma_list, D_input_dim_dict, train_eval):#utilityType):#input an entire sample set, or a batch sample set
        D_input_dict_list = []
        for df, data in df_data_dict.items():
            if self.isStacking and train_eval == "train":
                syn_D_input_list = []
            else:
                D_input_dict = OrderedDict()
            sample_strat_list = []
            sample_meanRet_strat_train_list = []
            sample_cov_strat_train_list = []
            sample_invCov_strat_train_list = []

            port_weights_train_list_dict = OrderedDict()
            performance_val_list_dict = OrderedDict()
            utility_val_list_dict = OrderedDict()
            for risk_gamma in risk_gamma_list:
                port_weights_train_list_dict[risk_gamma] = []
                performance_val_list_dict[risk_gamma] = []
                utility_val_list_dict[risk_gamma] = []
            for sample in data.split(1):
                if self.isStacking and train_eval == "train":
                    syn_D_input_per_sample_list = []
                sample = sample.reshape((sample.shape[1],sample.shape[2])).float().cuda()
 
                #train samples
                sample_strat_train = sample[:self.strat_train_days, :]#1
                sample_meanRet_strat_train = torch.mean(sample_strat_train, 0)#2
                sample_adj_strat_train = Variable(torch.add(sample_strat_train, -sample_meanRet_strat_train.repeat(self.strat_train_days, 1))) 
                sample_cov_strat_train = torch.div(torch.mm(sample_adj_strat_train.t(), sample_adj_strat_train), self.strat_train_days - 1)#3
                sample_invCov_strat_train = torch.inverse(torch.add((1 - self.shrink) * sample_cov_strat_train, self.shrink * torch.eye(self.mu_dim).cuda()))#4
                one = torch.ones(sample_invCov_strat_train.size()[0], 1).cuda()
                one_invCov = torch.mm(one.t(), sample_invCov_strat_train)
                one_invCov_one = torch.mm(one_invCov, one).item()
                one_invCov_ret = torch.mm(one_invCov, sample_meanRet_strat_train.view(-1, 1)).item()
                #eval samples
                sample_strat_val = sample[self.strat_train_days:,:]

                if train_eval == "eval":
                    randIndices_val = torch.randint(0, self.strat_val_days, (1,))
                    randIndices = torch.randint(0, self.strat_train_days + self.strat_val_days, (1,))

                if train_eval == "train":
                    sample_strat_list.append(sample)
                    if self.isStacking and ("ret" in D_input_dim_dict):
                        syn_D_input_per_sample_list.append(sample.view(1, -1))
                elif train_eval == "eval":
                    sample_strat_list.append(sample[randIndices,:])
                sample_meanRet_strat_train_list.append(sample_meanRet_strat_train.view(1, -1))
                if self.isStacking and train_eval == "train" and ("retEst" in D_input_dim_dict):
                    syn_D_input_per_sample_list.append(sample_meanRet_strat_train.view(1, -1))
                sample_cov_strat_train_list.append(sample_cov_strat_train.flatten().view(1, -1))
                if self.isStacking and train_eval == "train" and ("covEst" in D_input_dim_dict):
                    syn_D_input_per_sample_list.append(sample_cov_strat_train.flatten().view(1, -1))
                sample_invCov_strat_train_list.append(sample_invCov_strat_train.flatten().view(1, -1))            
                if self.isStacking and train_eval == "train" and ("invCovEst" in D_input_dim_dict):
                    syn_D_input_per_sample_list.append(sample_invCov_strat_train.flatten().view(1, -1))

                for risk_gamma in risk_gamma_list:
                    lamb = torch.div(one_invCov_ret - 2 * risk_gamma, one_invCov_one).item()
                    port_weights_train = torch.mm(sample_invCov_strat_train, torch.div(sample_meanRet_strat_train.view(-1,1) - lamb * one, 2 * risk_gamma))
                    port_weights_train_list_dict[risk_gamma].append(port_weights_train.t())#5
                    if self.isStacking and train_eval == "train" and ("portW" in D_input_dim_dict):
                        syn_D_input_per_sample_list.append(port_weights_train.t().view(1, -1))
                    performance_val = torch.mm(sample_strat_val, port_weights_train)
                    if train_eval == "train":
                        performance_val_list_dict[risk_gamma].append(performance_val)#6
                        if self.isStacking and ("per" in D_input_dim_dict):
                            syn_D_input_per_sample_list.append(performance_val.view(1, -1))
                    elif train_eval == "eval":
                        performance_val_list_dict[risk_gamma].append(performance_val[randIndices_val,:])#6
                    utility_val = performance_val - risk_gamma * performance_val * performance_val
                    if train_eval == "train" and ("utility" in D_input_dim_dict):
                        utility_val_list_dict[risk_gamma].append(utility_val)#7
                        if self.isStacking:
                            syn_D_input_per_sample_list.append(utility_val.view(1, -1))
                    elif train_eval == "eval":
                        utility_val_list_dict[risk_gamma].append(utility_val[randIndices_val,:])#7
                if self.isStacking and train_eval == "train":
                    syn_D_input_list.append(torch.cat(syn_D_input_per_sample_list, 1))    
            if self.isStacking and train_eval == "train":
                pass
            else:
                if "ret" in D_input_dim_dict:
                    D_input_dict["df" + str(df) + "_ret"] = torch.cat(sample_strat_list, 0)
                if "retEst" in D_input_dim_dict:
                    D_input_dict["df" + str(df) + "_retEst"] = torch.cat(sample_meanRet_strat_train_list, 0)
                if "covEst" in D_input_dim_dict:
                    D_input_dict["df" + str(df) + "_covEst"] = torch.cat(sample_cov_strat_train_list, 0)
                if "invCovEst" in D_input_dim_dict:
                    D_input_dict["df" + str(df) + "_invCovEst"] = torch.cat(sample_invCov_strat_train_list, 0)
                for risk_gamma in risk_gamma_list:
                    if "portW" in D_input_dim_dict:
                        D_input_dict["df" + str(df) + "_rg" + str(risk_gamma) + "_portW"] = torch.cat(port_weights_train_list_dict[risk_gamma], 0)
                    if "per" in D_input_dim_dict:
                        D_input_dict["df" + str(df) + "_rg" + str(risk_gamma) + "_per"] = torch.cat(performance_val_list_dict[risk_gamma], 0)
                    if "utility" in D_input_dim_dict:
                        D_input_dict["df" + str(df) + "_rg" + str(risk_gamma) + "_utility"] = torch.cat(utility_val_list_dict[risk_gamma], 0)
                D_input_dict_list.append(D_input_dict)
        if self.isStacking and train_eval == "train":
            syn_D_input = torch.cat(syn_D_input_list, 0)
            return syn_D_input
        else:
            return D_input_dict_list

    def computeDGLossName_RT(self):
        result_D_loss_name = []
        result_G_loss_name = []
        if self.isStacking:
            for lossName, discriminator in self.discriminator_dict.items():
                if lossName == "ret_per_utility":
                    for horizon in range(self.trainHorizonRange):
                        result_D_loss_name.append("D_loss_" + lossName + "_" + str(horizon))
                        result_G_loss_name.append("G_loss_" + lossName + "_" + str(horizon))
                elif lossName == "retEst_covEst_invCovEst_portW":
                    for horizon in range(self.trainHorizonRange - 1):
                        result_D_loss_name.append("D_loss_" + lossName + "_" + str(horizon))
                        result_G_loss_name.append("G_loss_" + lossName + "_" + str(horizon))
        else:
            for lossName, discriminator in self.discriminator_dict.items():
                if lossName in ["ret"]:
                    for horizon in range(self.trainHorizonRange):
                        result_D_loss_name.append("D_loss_" + lossName + "_" + str(horizon))
                        result_G_loss_name.append("G_loss_" + lossName + "_" + str(horizon))
                elif lossName in ["retEst", "covEst", "invCovEst"]:
                    for horizon in range(self.trainHorizonRange - 1):
                        result_D_loss_name.append("D_loss_" + lossName + "_" + str(horizon))
                        result_G_loss_name.append("G_loss_" + lossName + "_" + str(horizon))
                elif lossName in ["portW"]:
                    for horizon in range(self.trainHorizonRange - 1):
                        result_D_loss_name.append("D_loss_" + lossName + "_" + str(horizon) + "_" + str(self.train_risk_gamma))
                        result_G_loss_name.append("G_loss_" + lossName + "_" + str(horizon) + "_" + str(self.train_risk_gamma))
                elif lossName in ["per", "utility"]:
                    for horizon in range(self.trainHorizonRange):
                        result_D_loss_name.append("D_loss_" + lossName + "_" + str(horizon) + "_" + str(self.train_risk_gamma))
                        result_G_loss_name.append("G_loss_" + lossName + "_" + str(horizon) + "_" + str(self.train_risk_gamma))
        return result_D_loss_name + result_G_loss_name

    def computeStatsName_RT(self, worldName, D_input_dim_dict):
        resultMean = []
        resultVar = []
        for name, dim in D_input_dim_dict.items():
            if name in ["ret"]:
                for horizon in range(self.evalHorizonRange):
                    resultMean.extend([worldName + "_" + name + "_mean"] * dim)
                    resultVar.extend([worldName + "_" + name + "_var"] * dim)
            elif name in ["retEst", "covEst", "invCovEst"]:
                for horizon in range(self.evalHorizonRange - 1):
                    resultMean.extend([worldName + "_" + name + "_mean"] * dim)
                    resultVar.extend([worldName + "_" + name + "_var"] * dim)
            elif name in ["portW"]:
                for horizon in range(self.evalHorizonRange - 1):
                    for risk_gamma in self.eval_risk_gamma_list:
                        resultMean.extend([worldName + "_rg" + str(risk_gamma) + "_" + name + "_mean"] * dim)
                        resultVar.extend([worldName + "_rg" + str(risk_gamma) + "_" + name + "_var"] * dim)
            elif name in ["per", "utility"]:
                for horizon in range(self.evalHorizonRange):
                    for risk_gamma in self.eval_risk_gamma_list:
                        resultMean.extend([worldName + "_rg" + str(risk_gamma) + "_" + name + "_mean"] * dim)
                        resultVar.extend([worldName + "_rg" + str(risk_gamma) + "_" + name + "_var"] * dim)
        return resultMean + resultVar

    def computeDistName_RT(self, distName, D_input_dim_dict, train_eval):
        result = []
        for name, dim in D_input_dim_dict.items():
            if name == "ret":
                for horizon in range(self.evalHorizonRange):
                    result.append(train_eval + "_" + distName + "_" + name)
            elif name == "retEst" or name == "covEst" or name == "invCovEst":
                for horizon in range(self.evalHorizonRange - 1):
                    result.append(train_eval + "_" + distName + "_" + name)
            elif name == "portW":
                for horizon in range(self.evalHorizonRange - 1):
                    for risk_gamma in self.eval_risk_gamma_list:
                        result.append(train_eval + "_" + distName + "_rg" + str(risk_gamma) + "_" + name)
            elif name == "per" or name == "utility":
                for horizon in range(self.evalHorizonRange):
                    for risk_gamma in self.eval_risk_gamma_list:
                        result.append(train_eval + "_" + distName + "_rg" + str(risk_gamma) + "_" + name)
        return result

    #only for training phase evaluation
    def computeDGLoss_RT(self, batchDInput_dict, ganDInput_dict, featVecForwardList):
        result_D_loss = []
        result_G_loss = []
        
        
        for lossName, discriminator in self.discriminator_dict.items():
            if self.isStacking:
                if lossName == "ret_per_utility":
                    for horizon in range(self.trainHorizonRange):
                        G_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                        D_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1))) - G_loss_ind
                        result_G_loss.append(G_loss_ind.item())
                        result_D_loss.append(D_loss_ind.item())
                        if self.isPrintOpen:
                            print(lossName + ", " + str(horizon) + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
                elif lossName == "retEst_covEst_invCovEst_portW":
                    for horizon in range(self.trainHorizonRange - 1):
                        G_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                        D_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1))) - G_loss_ind
                        result_G_loss.append(G_loss_ind.item())
                        result_D_loss.append(D_loss_ind.item())
                        if self.isPrintOpen:
                            print(lossName + ", " + str(horizon) + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
            else:
                if lossName in ["ret"]:
                    for horizon in range(self.trainHorizonRange):
                        G_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                        D_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1))) - G_loss_ind
                        result_G_loss.append(G_loss_ind.item())
                        result_D_loss.append(D_loss_ind.item())
                        if self.isPrintOpen:
                            print(lossName + ", " + str(horizon) + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
                elif lossName in ["retEst", "covEst", "invCovEst"]:
                    for horizon in range(self.trainHorizonRange - 1):
                        G_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                        D_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1))) - G_loss_ind
                        result_G_loss.append(G_loss_ind.item())
                        result_D_loss.append(D_loss_ind.item())
                        if self.isPrintOpen:
                            print(lossName + ", " + str(horizon) + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
                elif lossName in ["portW"]:
                    for horizon in range(self.trainHorizonRange - 1):
                        G_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))
                        D_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1))) - G_loss_ind
                        result_G_loss.append(G_loss_ind.item())
                        result_D_loss.append(D_loss_ind.item())
                        if self.isPrintOpen:
                            print(lossName + ", " + str(horizon) + ", " + str(self.train_risk_gamma) + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
                elif lossName in ["per", "utility"]:
                    for horizon in range(self.trainHorizonRange):
                        G_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))
                        D_loss_ind = - self.horizonDecay ** horizon * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1))) - G_loss_ind
                        result_G_loss.append(G_loss_ind.item())
                        result_D_loss.append(D_loss_ind.item())
                        if self.isPrintOpen:
                            print(lossName + ", " + str(horizon) + ", " + str(self.train_risk_gamma) + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
        return result_D_loss + result_G_loss

    def computeStats_RT(self, D_input_dict):
        statsMean = []
        statsVar = []
#        for D_input_dict in D_input_dict_list:
        for name, data in D_input_dict.items():
            if name in ["ret"]:
                for horizon in range(self.evalHorizonRange):
                    mean = np.mean(data[horizon].data.cpu().numpy(), axis = 0)
                    var = np.var(data[horizon].data.cpu().numpy(), axis = 0)
                    if mean.shape == ():
                        statsMean.append(mean.item())
                        statsVar.append(var.item())
                    else:
                        statsMean.extend(mean.tolist())
                        statsVar.extend(var.tolist())
            elif name in ["retEst", "covEst", "invCovEst"]:
                for horizon in range(self.evalHorizonRange - 1):
                    mean = np.mean(data[horizon].data.cpu().numpy(), axis = 0)
                    var = np.var(data[horizon].data.cpu().numpy(), axis = 0)
                    if mean.shape == ():
                        statsMean.append(mean.item())
                        statsVar.append(var.item())
                    else:
                        statsMean.extend(mean.tolist())
                        statsVar.extend(var.tolist())
            elif name in ["portW"]:
                for horizon in range(self.evalHorizonRange - 1):
                    for risk_gamma in self.eval_risk_gamma_list:
                        mean = np.mean(data[horizon][risk_gamma].data.cpu().numpy(), axis = 0)
                        var = np.var(data[horizon][risk_gamma].data.cpu().numpy(), axis = 0)
                        if mean.shape == ():
                            statsMean.append(mean.item())
                            statsVar.append(var.item())
                        else:
                            statsMean.extend(mean.tolist())
                            statsVar.extend(var.tolist())
            elif name in ["per", "utility"]:
                for horizon in range(self.evalHorizonRange):
                    for risk_gamma in self.eval_risk_gamma_list:
                        mean = np.mean(data[horizon][risk_gamma].data.cpu().numpy(), axis = 0)
                        var = np.var(data[horizon][risk_gamma].data.cpu().numpy(), axis = 0)
                        if mean.shape == ():
                            statsMean.append(mean.item())
                            statsVar.append(var.item())
                        else:
                            statsMean.extend(mean.tolist())
                            statsVar.extend(var.tolist())
        return statsMean + statsVar

    def computeWass_RT(self, D_input_dict, gan_input_dict):
        result = []
        for D_input_dict_item, gan_input_dict_item in zip(D_input_dict.items(), gan_input_dict.items()):
            if D_input_dict_item[0] in ["ret"]:
                for horizon in range(self.evalHorizonRange):
                    num_samples = D_input_dict_item[1][horizon].shape[0]
                    sampleD = D_input_dict_item[1][horizon]
                    sampleGAN = gan_input_dict_item[1][horizon]
                    uniform1 = np.ones((num_samples, )) / num_samples
                    uniform2 = np.ones((num_samples, )) / num_samples
                    M = ot.dist(sampleD.cpu().detach().numpy(), sampleGAN.cpu().detach().numpy())
                    M /= M.max()
                    wass = ot.emd2(uniform1, uniform2, M, numItermax=1000000)
                    result.append(wass)
            elif D_input_dict_item[0] in ["retEst", "covEst", "invCovEst"]:
                for horizon in range(self.evalHorizonRange - 1):
                    num_samples = D_input_dict_item[1][horizon].shape[0]
                    sampleD = D_input_dict_item[1][horizon]
                    sampleGAN = gan_input_dict_item[1][horizon]
                    uniform1 = np.ones((num_samples, )) / num_samples
                    uniform2 = np.ones((num_samples, )) / num_samples
                    M = ot.dist(sampleD.cpu().detach().numpy(), sampleGAN.cpu().detach().numpy())
                    M /= M.max()
                    wass = ot.emd2(uniform1, uniform2, M, numItermax=1000000)
                    result.append(wass)
            elif D_input_dict_item[0] in ["portW"]:
                for horizon in range(self.evalHorizonRange - 1):
                    for risk_gamma in self.eval_risk_gamma_list:
                        num_samples = D_input_dict_item[1][horizon][risk_gamma].shape[0]
                        sampleD = D_input_dict_item[1][horizon][risk_gamma]
                        sampleGAN = gan_input_dict_item[1][horizon][self.train_risk_gamma]
                        uniform1 = np.ones((num_samples, )) / num_samples
                        uniform2 = np.ones((num_samples, )) / num_samples
                        M = ot.dist(sampleD.cpu().detach().numpy(), sampleGAN.cpu().detach().numpy())
                        M /= M.max()
                        wass = ot.emd2(uniform1, uniform2, M, numItermax=1000000)
                        result.append(wass)
            elif D_input_dict_item[0] in ["per", "utility"]:
                for horizon in range(self.evalHorizonRange):
                    for risk_gamma in self.eval_risk_gamma_list:
                        num_samples = D_input_dict_item[1][horizon][risk_gamma].shape[0]
                        sampleD = D_input_dict_item[1][horizon][risk_gamma]
                        sampleGAN = gan_input_dict_item[1][horizon][self.train_risk_gamma]
                        uniform1 = np.ones((num_samples, )) / num_samples
                        uniform2 = np.ones((num_samples, )) / num_samples
                        M = ot.dist(sampleD.cpu().detach().numpy(), sampleGAN.cpu().detach().numpy())
                        M /= M.max()
                        wass = ot.emd2(uniform1, uniform2, M, numItermax=1000000)
                        result.append(wass)

        return result

    def computeDGLossName(self):
        result_D_loss_name = []
        result_G_loss_name = []
        for lossName, discriminator in self.discriminator_dict.items():
            result_D_loss_name.append("D_loss_" + lossName)
            result_G_loss_name.append("G_loss_" + lossName)
        return result_D_loss_name + result_G_loss_name

    def computeStatsName(self, worldName, df_list, risk_gamma_list, D_input_dim_dict):
        resultMean = []
        resultVar = []
        for df in df_list:
            for name, dim in D_input_dim_dict.items():
                if name == "ret" or name == "retEst" or name == "covEst" or name == "invCovEst":
                    resultMean.extend([worldName + "_df" + str(df) + "_" + name + "_mean"] * dim)
                    resultVar.extend([worldName + "_df" + str(df) + "_" + name + "_var"] * dim)
            for risk_gamma in risk_gamma_list:
                for name, dim in D_input_dim_dict.items():
                    if name == "portW" or name == "per" or name == "utility":
                        resultMean.extend([worldName + "_df" + str(df) + "_rg" + str(risk_gamma) + "_" + name + "_mean"] * dim)
                        resultVar.extend([worldName + "_df" + str(df) + "_rg" + str(risk_gamma) + "_" + name + "_var"] * dim)
        return resultMean + resultVar

    def computeDistName(self, distName, df_list, risk_gamma_list, D_input_dim_dict, train_eval):
        result = []
        for df in df_list:
            for name, dim in D_input_dim_dict.items():
                if name == "ret" or name == "retEst" or name == "covEst" or name == "invCovEst":
                    result.append(train_eval + "_" + distName + "_" + "df" + str(df) + "_" + name)
            for risk_gamma in risk_gamma_list:
                for name, dim in D_input_dim_dict.items():
                    if name == "portW" or name == "per" or name == "utility":
                        result.append(train_eval + "_" + distName + "_" + "df" + str(df) + "_rg" + str(risk_gamma) + "_" + name)
        return result

    def computeDGLoss(self, batchDInput_dict, ganDInput_dict):
        result_D_loss = []
        result_G_loss = []
        
        for lossName, discriminator in self.discriminator_dict.items():
            if lossName in ["ret", "retEst", "covEst", "invCovEst"]:
                G_loss_ind = -torch.mean(discriminator(ganDInput_dict["df" + str(self.train_df) + "_" + lossName]))
                D_loss_ind = -torch.mean(discriminator(batchDInput_dict["df" + str(self.train_df) + "_" + lossName])) - G_loss_ind
                result_G_loss.append(G_loss_ind.item())
                result_D_loss.append(D_loss_ind.item())
                if self.isPrintOpen:
                    print("df" + str(self.train_df) + "_" + lossName + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
            elif lossName in ["portW", "per", "utility"]:
                G_loss_ind = -torch.mean(discriminator(ganDInput_dict["df" + str(self.train_df) + "_rg" + str(self.train_risk_gamma) + "_" + lossName]))
                D_loss_ind = -torch.mean(discriminator(batchDInput_dict["df" + str(self.train_df) + "_rg" + str(self.train_risk_gamma) + "_" + lossName])) - G_loss_ind
                result_G_loss.append(G_loss_ind.item())
                result_D_loss.append(D_loss_ind.item())
                if self.isPrintOpen:
                    print("df" + str(self.train_df) + "_rg" + str(self.train_risk_gamma) + "_" + lossName + ", D_loss: ", D_loss_ind.item(), ", G_loss: ", G_loss_ind.item())
        return result_D_loss + result_G_loss

    def computeStats(self, D_input_dict_list):
        statsMean = []
        statsVar = []
        for D_input_dict in D_input_dict_list:
            for name, data in D_input_dict.items():
                mean = np.mean(data.data.cpu().numpy(), axis = 0)
                var = np.var(data.data.cpu().numpy(), axis = 0)
                if mean.shape == ():
                    statsMean.append(mean.item())
                    statsVar.append(var.item())
                else:
                    statsMean.extend(mean.tolist())
                    statsVar.extend(var.tolist())
        return statsMean + statsVar

    def computeWass(self, D_input_dict_list, gan_input_dict):
        result = []
        for D_input_dict in D_input_dict_list:
            for D_input_dict_item, gan_input_dict_item in zip(D_input_dict.items(), gan_input_dict.items()):
                sampleD = D_input_dict_item[1].view(-1, D_input_dict_item[1].shape[1])
                sampleGAN = gan_input_dict_item[1].view(-1, D_input_dict_item[1].shape[1])
                uniform = np.ones((self.train_num_samples, )) / self.train_num_samples
                M = ot.dist(sampleD.cpu().detach().numpy(), sampleGAN.cpu().detach().numpy())
                M /= M.max()
                wass = ot.emd2(uniform, uniform, M)
                result.append(wass)
        return result

    def computeDInputHistDataFromHist(self, batch_data, D_input_dim_dict, horizonRange):
        #targets, R.V.(the first future date is day 0)
        tgtDailyRetList = []
        
        tgtDailyPerDictList = []
        tgtDailyUtilityDictList = []

        tgtDailyRetEstList = []
        tgtDailyCovEstList = []
        tgtDailyInvCovEstList = []
        tgtDailyPortWDictList = []

        if self.isStacking:
            tgtDailyRetPerUtilityList = []
            tgtDailyRetEstCovEstInvCovEstPortWList = []

        for horizon in range(horizonRange):
            #part1: fundamental variable for current day
            tgtDailyRet = Variable(batch_data[:, self.tgtDailyRetColIndBEListList[horizon][0] : self.tgtDailyRetColIndBEListList[horizon][1]].float().cuda())
            tgtDailyRetList.append(tgtDailyRet)

            #part2: realized performance/utility for prev day
            tgtDailyPerDict = OrderedDict()
            tgtDailyUtilityDict = OrderedDict()
            for risk_gamma in self.eval_risk_gamma_list:
                tgtDailyPerDict[risk_gamma] = batch_data[:, self.tgtDailyPerColIndBEListDictList[horizon][risk_gamma][0] : self.tgtDailyPerColIndBEListDictList[horizon][risk_gamma][1]].float().cuda()
                tgtDailyUtilityDict[risk_gamma] = batch_data[:, self.tgtDailyUtilityColIndBEListDictList[horizon][risk_gamma][0] : self.tgtDailyUtilityColIndBEListDictList[horizon][risk_gamma][1]].float().cuda()
            tgtDailyPerDictList.append(tgtDailyPerDict)
            tgtDailyUtilityDictList.append(tgtDailyUtilityDict)
        
            if self.isStacking:
                tgtDailyRetPerUtility = []
                if "ret" in self.D_input_dim_train_dict:
                    tgtDailyRetPerUtility.append(tgtDailyRet)
                if "per" in self.D_input_dim_train_dict:
                    tgtDailyRetPerUtility.append(tgtDailyPerDict[self.train_risk_gamma])
                if "utility" in self.D_input_dim_train_dict:
                    tgtDailyRetPerUtility.append(tgtDailyUtilityDict[self.train_risk_gamma])
                if len(tgtDailyRetPerUtility) > 0:
                    tgtDailyRetPerUtilityList.append(torch.cat(tgtDailyRetPerUtility, 1))
                
       
           
            #part3: estimators for next day
            if horizon < horizonRange - 1:
                tgtDailyRetEst = Variable(batch_data[:, self.tgtDailyRetEstColIndBEListList[horizon][0] : self.tgtDailyRetEstColIndBEListList[horizon][1]].float().cuda())
                tgtDailyCovEst = Variable(batch_data[:, self.tgtDailyCovEstColIndBEListList[horizon][0] : self.tgtDailyCovEstColIndBEListList[horizon][1]].float().cuda())
                tgtDailyInvCovEst = Variable(batch_data[:, self.tgtDailyInvCovEstColIndBEListList[horizon][0] : self.tgtDailyInvCovEstColIndBEListList[horizon][1]].float().cuda())
                tgtDailyPortWDict = OrderedDict()
                for risk_gamma in self.eval_risk_gamma_list:
                    tgtDailyPortWDict[risk_gamma] = batch_data[:, self.tgtDailyPortWColIndBEListDictList[horizon][risk_gamma][0] : self.tgtDailyPortWColIndBEListDictList[horizon][risk_gamma][1]].float().cuda()
                tgtDailyRetEstList.append(tgtDailyRetEst)
                tgtDailyCovEstList.append(tgtDailyCovEst)
                tgtDailyInvCovEstList.append(tgtDailyInvCovEst)
                tgtDailyPortWDictList.append(tgtDailyPortWDict)

                if self.isStacking:
                    tgtDailyRetEstCovEstInvCovEstPortW = []
                    if "retEst" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyRetEst)
                    if "covEst" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyCovEst)
                    if "invCovEst" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyInvCovEst)
                    if "portW" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyPortWDict[self.train_risk_gamma])
                    if len(tgtDailyRetEstCovEstInvCovEstPortW) > 0:
                        tgtDailyRetEstCovEstInvCovEstPortWList.append(torch.cat(tgtDailyRetEstCovEstInvCovEstPortW, 1))
            

        syn_D_input = OrderedDict()

        if "ret" in D_input_dim_dict:
            syn_D_input["ret"] = tgtDailyRetList

        if "per" in D_input_dim_dict:
            syn_D_input["per"] = tgtDailyPerDictList
        if "utility" in D_input_dim_dict:
            syn_D_input["utility"] = tgtDailyUtilityDictList

        if "retEst" in D_input_dim_dict:
            syn_D_input["retEst"] = tgtDailyRetEstList
        if "covEst" in D_input_dim_dict:
            syn_D_input["covEst"] = tgtDailyCovEstList
        if "invCovEst" in D_input_dim_dict:
            syn_D_input["invCovEst"] = tgtDailyInvCovEstList
        if "portW" in D_input_dim_dict:
            syn_D_input["portW"] = tgtDailyPortWDictList

        if self.isStacking:
            if len(tgtDailyRetPerUtilityList) > 0:
                syn_D_input["ret_per_utility"] = tgtDailyRetPerUtilityList
            if len(tgtDailyRetEstCovEstInvCovEstPortWList) > 0:
                syn_D_input["retEst_covEst_invCovEst_portW"] = tgtDailyRetEstCovEstInvCovEstPortWList
        
        return syn_D_input       

    #only for training for real data scenario
    #include generator forward step
    def computeDInputHistDataFromGenerator(self, batch_data, one, diagMat, risk_gamma_list, D_input_dim_dict, horizonRange):
        sample_size = batch_data.shape[0]
        #features
        featVecSymbolListForwardList = []#outer index is forward horizons
        featVecForwardList = []
        featDailyRetVecEMADictForwardList = []

        featVecSymbolList = []
        featDailyRetSymbolListEMADict = OrderedDict()
        for halfLife in self.halfLifeList:
            featDailyRetSymbolListEMADict[halfLife] = []
        featDailyRetVecEMADict = OrderedDict()
        for etfName in self.targetEtfNameList:
            featList = []
            #featDailyRet
            featDailyRetSymbol = Variable(batch_data[:, self.featColIndBEListDailyRetSymbolDict[etfName][0] : self.featColIndBEListDailyRetSymbolDict[etfName][1]].float().cuda())
            featList.append(featDailyRetSymbol)

            #featDailyRetEMA
            featDailyRetEMADict = OrderedDict()
            for halfLife in self.halfLifeList:
                featDailyRetEMADict[halfLife] = Variable(batch_data[:, self.featColIndBEListDailyRetEMADictSymbolDict[etfName][halfLife][0] : self.featColIndBEListDailyRetEMADictSymbolDict[etfName][halfLife][1]].float().cuda())
                featDailyRetSymbolListEMADict[halfLife].append(featDailyRetEMADict[halfLife])
                featList.append(featDailyRetEMADict[halfLife])
            featVecSymbolList.append(torch.cat(featList, 1))
        for halfLife in self.halfLifeList:
            featDailyRetVecEMADict[halfLife] = torch.cat(featDailyRetSymbolListEMADict[halfLife], 1)
        featDailyRetVecEMADictForwardList.append(featDailyRetVecEMADict)
        featVecSymbolListForwardList.append(featVecSymbolList)
        featVecForwardList.append(torch.cat(featVecSymbolList, 1))

        #targets, R.V., horizon 0 is first future date
        tgtDailyRetList = []
        
        tgtDailyPerDictList = []
        tgtDailyUtilityDictList = []

        tgtDailyRetEstList = []
        tgtDailyCovEstList = []
        tgtDailyInvCovEstList = []
        tgtDailyPortWDictList = []

        if self.isStacking:
            tgtDailyRetPerUtilityList = []
            tgtDailyRetEstCovEstInvCovEstPortWList = []

        warmUpTgtDailyPortW = OrderedDict()
        for risk_gamma in risk_gamma_list:
            warmUpTgtDailyPortW[risk_gamma] = Variable(batch_data[:, self.warmUpTgtDailyPortWColIndBEListDict[risk_gamma][0]: self.warmUpTgtDailyPortWColIndBEListDict[risk_gamma][1]].float().cuda())
        warmUpTgtDailyRetEst = Variable(batch_data[:, self.warmUpTgtDailyRetEstColIndBEList[0]: self.warmUpTgtDailyRetEstColIndBEList[1]].float().cuda())
        warmUpTgtDailyCovEst = Variable(batch_data[:, self.warmUpTgtDailyCovEstColIndBEList[0]: self.warmUpTgtDailyCovEstColIndBEList[1]].float().cuda())

        estDecay = math.exp(math.log(0.5) / self.estHalfLife)        

        for horizon in range(horizonRange):
            #targets

            #part1: fudamental quantity for curr day
            z = Variable(torch.randn(sample_size, self.z_dim).cuda())

            if self.hasAutoRegBias:
                augFeatSymbolList = []
                for i in range(len(self.targetEtfNameList)):
                    augFeatSymbolList.append(torch.cat((z, featVecSymbolListForwardList[horizon][i]), 1))
                tgtDailyRet = self.generator(augFeatSymbolList)
            else:
                augFeatSymbolList = []
                for i in range(len(self.targetEtfNameList)):
                    augFeatSymbolList.append(torch.cat((z, featVecSymbolListForwardList[0][i]), 1))
                tgtDailyRet = self.generator(augFeatSymbolList)
            tgtDailyRetList.append(tgtDailyRet)#days * mu_dim
            

            #part2: realized performance/utility for prev day
            tgtDailyPerDict = OrderedDict()
            if horizon == 0:
                for risk_gamma in risk_gamma_list:
                    tgtDailyPerDict[risk_gamma] = torch.matmul(torch.unsqueeze(warmUpTgtDailyPortW[risk_gamma], 2).transpose(1,2), torch.unsqueeze(tgtDailyRet, 2)).view(sample_size, -1)
                tgtDailyPerDictList.append(tgtDailyPerDict)
            else:
                for risk_gamma in risk_gamma_list:
                    tgtDailyPerDict[risk_gamma] = torch.matmul(torch.unsqueeze(tgtDailyPortWDictList[horizon - 1][risk_gamma], 2).transpose(1,2), torch.unsqueeze(tgtDailyRet, 2)).view(sample_size, -1)
                tgtDailyPerDictList.append(tgtDailyPerDict)

            tgtDailyUtilityDict = OrderedDict()
            for risk_gamma in risk_gamma_list:
                tgtDailyUtilityDict[risk_gamma] = tgtDailyPerDict[risk_gamma] - risk_gamma * tgtDailyPerDict[risk_gamma] ** 2
            tgtDailyUtilityDictList.append(tgtDailyUtilityDict)

            if self.isStacking:
                tgtDailyRetPerUtility = []
                if "ret" in self.D_input_dim_train_dict:
                    tgtDailyRetPerUtility.append(tgtDailyRet)
                if "per" in self.D_input_dim_train_dict:
                    tgtDailyRetPerUtility.append(tgtDailyPerDict[self.train_risk_gamma])
                if "utility" in self.D_input_dim_train_dict:
                    tgtDailyRetPerUtility.append(tgtDailyUtilityDict[self.train_risk_gamma])
                if len(tgtDailyRetPerUtility) > 0:
                    tgtDailyRetPerUtilityList.append(torch.cat(tgtDailyRetPerUtility, 1)) 
           
            if horizon < horizonRange - 1:
                #part3: estimators for next day
                if horizon == 0:
                    tgtDailyRetEst = warmUpTgtDailyRetEst * estDecay + tgtDailyRet * (1 - estDecay)
                else:
                    tgtDailyRetEst = tgtDailyRetEstList[horizon - 1] * estDecay + tgtDailyRet * (1 - estDecay)
                tgtDailyRetEstList.append(tgtDailyRetEst)

                if horizon == 0:
                    tgtDailyCovEst = warmUpTgtDailyCovEst * estDecay + torch.matmul(torch.unsqueeze(tgtDailyRet, 2), torch.unsqueeze(tgtDailyRet, 1)).view(sample_size, -1) * (1 - estDecay)
                else:
                    tgtDailyCovEst = tgtDailyCovEstList[horizon - 1] * estDecay + torch.matmul(torch.unsqueeze(tgtDailyRet, 2), torch.unsqueeze(tgtDailyRet, 1)).view(sample_size, -1) * (1 - estDecay)
                tgtDailyCovEstList.append(tgtDailyCovEst)

                tgtDailyInvCovEst = torch.inverse(tgtDailyCovEst.view(sample_size, self.numTargetEtf, self.numTargetEtf) * (1 - self.shrink) +  tgtDailyCovEst.view(sample_size, self.numTargetEtf, self.numTargetEtf) * diagMat * self.shrink)
                tgtDailyInvCovEstList.append(tgtDailyInvCovEst.contiguous().view(sample_size, -1))

                tgtDailyPortWDict = OrderedDict()
                for risk_gamma in risk_gamma_list:
                    tgtDailyPortWDict[risk_gamma] = self.compPortWTorch(tgtDailyRetEst, tgtDailyInvCovEst, one, risk_gamma)
                tgtDailyPortWDictList.append(tgtDailyPortWDict)

                if self.isStacking:
                    tgtDailyRetEstCovEstInvCovEstPortW = []
                    if "retEst" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyRetEst)
                    if "covEst" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyCovEst)
                    if "invCovEst" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyInvCovEst.contiguous().view(sample_size, -1))
                    if "portW" in self.D_input_dim_train_dict:
                        tgtDailyRetEstCovEstInvCovEstPortW.append(tgtDailyPortWDict[self.train_risk_gamma])
                    if len(tgtDailyRetEstCovEstInvCovEstPortW) > 0:
                        tgtDailyRetEstCovEstInvCovEstPortWList.append(torch.cat(tgtDailyRetEstCovEstInvCovEstPortW, 1))

                if not self.hasAutoRegBias:
                    continue
                #part4: features for next day
                featVecSymbolList = []
                featDailyRetSymbolListEMADict = OrderedDict()
                for halfLife in self.halfLifeList:
                    featDailyRetSymbolListEMADict[halfLife] = []
                featDailyRetVecEMADict = OrderedDict()
                #physical meaning features
                for i in range(len(self.targetEtfNameList)):
                    featList = []
                    #featDailyRet
                    featDailyRetSymbol = tgtDailyRet[:, i].reshape(-1,1)
                    featList.append(featDailyRetSymbol)

                    #featDailyRetEMA
                    featDailyRetEMADict = OrderedDict()
                    for halfLife in self.halfLifeList:
                        decay = math.exp(math.log(0.5) / halfLife)
                        featDailyRetEMADict[halfLife] = featDailyRetVecEMADictForwardList[horizon][halfLife][:, i].reshape(-1,1) * decay + featDailyRetSymbol * (1 - decay)
                        featDailyRetSymbolListEMADict[halfLife].append(featDailyRetEMADict[halfLife])
                        featList.append(featDailyRetEMADict[halfLife])
                    featVecSymbolList.append(torch.cat(featList, 1))
                for halfLife in self.halfLifeList:
                    featDailyRetVecEMADict[halfLife] = torch.cat(featDailyRetSymbolListEMADict[halfLife], 1)
                featDailyRetVecEMADictForwardList.append(featDailyRetVecEMADict)
                featVecSymbolListForwardList.append(featVecSymbolList)
                featVecForwardList.append(torch.cat(featVecSymbolList, 1))

        syn_D_input = OrderedDict()
        if "ret" in D_input_dim_dict:
            syn_D_input["ret"] = tgtDailyRetList
        
        if "per" in D_input_dim_dict:
            syn_D_input["per"] = tgtDailyPerDictList
        if "utility" in D_input_dim_dict:
            syn_D_input["utility"] = tgtDailyUtilityDictList

        if "retEst" in D_input_dim_dict:
            syn_D_input["retEst"] = tgtDailyRetEstList
        if "covEst" in D_input_dim_dict:
            syn_D_input["covEst"] = tgtDailyCovEstList
        if "invCovEst" in D_input_dim_dict:
            syn_D_input["invCovEst"] = tgtDailyInvCovEstList
        if "portW" in D_input_dim_dict:
            syn_D_input["portW"] = tgtDailyPortWDictList

        if self.isStacking:
            if len(tgtDailyRetPerUtilityList) > 0:
                syn_D_input["ret_per_utility"] = tgtDailyRetPerUtilityList
            if len(tgtDailyRetEstCovEstInvCovEstPortWList) > 0:
                syn_D_input["retEst_covEst_invCovEst_portW"] = tgtDailyRetEstCovEstInvCovEstPortWList

        return syn_D_input, featVecForwardList



    def train_CGAN_RT(self):
        trainloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        one = torch.ones(self.batch_size, self.numTargetEtf).cuda()
        diagMat = torch.eye(self.numTargetEtf).cuda().reshape((1, self.numTargetEtf, self.numTargetEtf)).repeat(self.batch_size, 1, 1)
        one_eval = torch.ones(self.eval_num_samples, self.numTargetEtf).cuda()
        diagMat_eval = torch.eye(self.numTargetEtf).cuda().reshape((1, self.numTargetEtf, self.numTargetEtf)).repeat(self.eval_num_samples, 1, 1)
        one_train = torch.ones(self.train_num_samples, self.numTargetEtf).cuda()
        diagMat_train = torch.eye(self.numTargetEtf).cuda().reshape((1, self.numTargetEtf, self.numTargetEtf)).repeat(self.train_num_samples, 1, 1)
        for epoch in range(self.n_epoch):
            for i, batch_data in enumerate(trainloader):
                if i % (self.n_critics + self.n_actors) < self.n_critics:
                    #forward train world samples
                    batchDInput_dict = self.computeDInputHistDataFromHist(batch_data, self.D_input_dim_train_dict, self.trainHorizonRange)

                    #forward fake world samples
                    ganDInput_dict, featVecForwardList = self.computeDInputHistDataFromGenerator(batch_data, one, diagMat, [self.train_risk_gamma], self.D_input_dim_train_dict, self.trainHorizonRange)
                    
                    #forward loss computation
                    #include discriminator forward step
                    D_loss = 0
                    for lossName, discriminator in self.discriminator_dict.items():
                        if self.isStacking:
                            if lossName == "ret_per_utility":
                                for horizon in range(self.trainHorizonRange):
                                    D_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                                    D_loss += self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                            elif lossName == "retEst_covEst_invCovEst_portW":
                                for horizon in range(self.trainHorizonRange - 1):
                                    D_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                                    D_loss += self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                        else:
                            if lossName in ["ret"]:
                                for horizon in range(self.trainHorizonRange):
                                    D_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                                    D_loss += self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                            elif lossName in ["retEst", "covEst", "invCovEst"]:
                                for horizon in range(self.trainHorizonRange - 1):
                                    D_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                                    D_loss += self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                            elif lossName in ["portW"]:
                                for horizon in range(self.trainHorizonRange - 1):
                                    D_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))
                                    D_loss += self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))
                            elif lossName in ["per", "utility"]:
                                for horizon in range(self.trainHorizonRange):
                                    D_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((batchDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))
                                    D_loss += self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))

                    #backward discriminator gradients update
                    D_loss.backward()
                    for lossName, discriminator_solver in self.discriminator_solver_dict.items():#note that this gradient is discriminator by discriminator
                        discriminator_solver.step()

		    #discriminator Weight clipping
                    for lossName, discriminator in self.discriminator_dict.items():
                        for p in discriminator.parameters():
                            p.data.clamp_(self.clip_lb, self.clip_ub)

		    # Housekeeping - reset gradient
                    self.generator.zero_grad()
                    for lossName, discriminator in self.discriminator_dict.items():
                        discriminator.zero_grad()
            
                else:
                    #forward fake world samples
                    ganDInput_dict, featVecForwardList = self.computeDInputHistDataFromGenerator(batch_data, one, diagMat, [self.train_risk_gamma], self.D_input_dim_train_dict, self.trainHorizonRange)

                    G_loss = 0
                    #include discriminator forward step
                    for lossName, discriminator in self.discriminator_dict.items():
                        if self.isStacking:
                            if lossName == "ret_per_utility":
                                for horizon in range(self.trainHorizonRange):
                                    G_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                            elif lossName == "retEst_covEst_invCovEst_portW":
                                for horizon in range(self.trainHorizonRange - 1):
                                    G_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                        else:
                            if lossName in ["ret"]:
                                for horizon in range(self.trainHorizonRange):
                                    G_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                            elif lossName in ["retEst", "covEst", "invCovEst"]:
                                for horizon in range(self.trainHorizonRange - 1):
                                    G_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon], featVecForwardList[0]), 1)))
                            elif lossName in ["portW"]:
                                for horizon in range(self.trainHorizonRange - 1):
                                    G_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))
                            elif lossName in ["per", "utility"]:
                                for horizon in range(self.trainHorizonRange):
                                    G_loss -= self.horizonDecay ** horizon * self.loss_weight_dict[lossName] * torch.mean(discriminator(torch.cat((ganDInput_dict[lossName][horizon][self.train_risk_gamma], featVecForwardList[0]), 1)))

                    #backward generator gradients update
                    G_loss.backward()#retain_graph = True)
                    self.generator_solver.step()

		    # Housekeeping - reset gradient
                    self.generator.zero_grad()
                    for lossName, discriminator in self.discriminator_dict.items():
                        discriminator.zero_grad()

            #eval step
            if epoch % self.evalFreq == 0:
                ganDInput_dict_train, featVecForwardList_train = self.computeDInputHistDataFromGenerator(self.train_data, one_train, diagMat_train, [self.train_risk_gamma], self.D_input_dim_eval_dict, self.evalHorizonRange)
                ganDInput_dict_eval, featVecForwardList_eval = self.computeDInputHistDataFromGenerator(self.eval_data, one_eval, diagMat_eval, self.eval_risk_gamma_list, self.D_input_dim_eval_dict, self.evalHorizonRange)#this one is tricky, since we are gonna to evaluate on training set for different gan loss
            

                if self.isPrintOpen:
                    print("Iter: ", epoch)
                    print("D_loss: ", D_loss.data.cpu().numpy(), "G_loss: ", G_loss.data.cpu().numpy())
                
                csv.writer(self.output_fakeWorld_file).writerow([epoch, D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy()] + self.computeDGLoss_RT(self.eval_D_input_dict_RT, ganDInput_dict_eval, featVecForwardList_eval) + \
self.computeWass_RT(self.train_D_input_dict_RT, ganDInput_dict_train) + self.computeWass_RT(self.eval_D_input_dict_RT, ganDInput_dict_eval))
                output_generator_file = open('out/' + self.dateToday + '/' + str(self.subFolderID) + '/generator_' + self.fileNamePostfix + "_" + str(epoch) + ".pt", 'wb')
                torch.save({"generator_state_dict": self.generator.state_dict()}, output_generator_file)
               
        self.output_fakeWorld_file.close()

    def trainGAN(self):
        trainloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.n_epoch):
            for i, batch_data in enumerate(trainloader):
                if i % (self.n_critics + self.n_actors) < self.n_critics:
                    #forward train world samples
                    batch_data = batch_data.reshape((self.batch_size, self.strat_train_days + self.strat_val_days, self.mu_dim))
                    df_data_dict = OrderedDict()
                    df_data_dict[self.train_df] = batch_data
                    if self.isStacking:
                        syn_batchDInput = self.computeDInput(df_data_dict, [self.train_risk_gamma], self.D_input_dim_train_dict, "train")
                    else: 
                        batchDInput_dict = self.computeDInput(df_data_dict, [self.train_risk_gamma], self.D_input_dim_train_dict, "train")[0]

                    #forward fake world samples
                    z = Variable(torch.randn(self.batch_size * (self.strat_train_days + self.strat_val_days), self.z_dim).cuda())
                    if self.isRealData:
                        G_data = self.generator(torch.cat((z, batch_data), 1))
                    else:
                        G_data = self.generator(z)
                    G_data = G_data.reshape((self.batch_size, self.strat_train_days + self.strat_val_days, self.mu_dim))
                    df_data_dict = OrderedDict()
                    df_data_dict[self.train_df] = G_data
                    if self.isStacking:
                        syn_ganDInput = self.computeDInput(df_data_dict, [self.train_risk_gamma], self.D_input_dim_train_dict, "train")
                    else:
                        ganDInput_dict = self.computeDInput(df_data_dict, [self.train_risk_gamma], self.D_input_dim_train_dict, "train")[0]
                    
                    #forward loss computation
                    D_loss = 0
                    if self.isStacking:
                        D_loss = torch.mean(self.discriminator(syn_ganDInput)) - torch.mean(self.discriminator(syn_batchDInput))
                    else:
                        for lossName, discriminator in self.discriminator_dict.items():
                            if lossName in ["ret", "retEst", "covEst", "invCovEst"]:
                                D_loss -= self.loss_weight_dict[lossName] * torch.mean(discriminator(batchDInput_dict["df" + str(self.train_df) + "_" + lossName]))
                                D_loss += self.loss_weight_dict[lossName] * torch.mean(discriminator(ganDInput_dict["df" + str(self.train_df) + "_" + lossName]))
                            elif lossName in ["portW", "per", "utility"]:
                                D_loss -= self.loss_weight_dict[lossName] * torch.mean(discriminator(batchDInput_dict["df" + str(self.train_df) + "_rg" + str(self.train_risk_gamma) + "_" + lossName]))
                                D_loss += self.loss_weight_dict[lossName] * torch.mean(discriminator(ganDInput_dict["df" + str(self.train_df) + "_rg" + str(self.train_risk_gamma) + "_" + lossName]))

                    #backward discriminator gradients update
                    D_loss.backward()
                    if self.isStacking:
                        self.discriminator_solver.step()
                    else:
                        for lossName, discriminator_solver in self.discriminator_solver_dict.items():
                            discriminator_solver.step()

		    #discriminator Weight clipping
                    if self.isStacking:
                        for p in self.discriminator.parameters():
                            p.data.clamp_(self.clip_lb, self.clip_ub)
                    else:
                        for lossName, discriminator in self.discriminator_dict.items():
                            for p in discriminator.parameters():
                                p.data.clamp_(self.clip_lb, self.clip_ub)

		    # Housekeeping - reset gradient
                    self.generator.zero_grad()
                    if self.isStacking:
                        self.discriminator.zero_grad()
                    else:
                        for lossName, discriminator in self.discriminator_dict.items():
                            discriminator.zero_grad()
            
                else:
                    #forward fake world samples
                    z = Variable(torch.randn(self.batch_size * (self.strat_train_days + self.strat_val_days), self.z_dim).cuda())
                    if self.isRealData:
                        G_data = self.generator(torch.cat((z, batch_data), 1))
                    else:
                        G_data = self.generator(x)
                    G_data = G_data.reshape((self.batch_size, self.strat_train_days + self.strat_val_days, self.mu_dim))
                    df_data_dict = OrderedDict()
                    df_data_dict[self.train_df] = G_data
                    if self.isStacking:
                        syn_ganDInput = self.computeDInput(df_data_dict, [self.train_risk_gamma], self.D_input_dim_train_dict, "train")
                    else:
                        ganDInput_dict = self.computeDInput(df_data_dict, [self.train_risk_gamma], self.D_input_dim_train_dict, "train")[0]
                    G_loss = 0
                    if self.isStacking:
                        G_loss -= torch.mean(self.discriminator(syn_ganDInput))
                    else:
                        for lossName, discriminator in self.discriminator_dict.items():
                            if lossName in ["ret", "retEst", "covEst", "invCovEst"]:
                                G_loss -= self.loss_weight_dict[lossName] * torch.mean(discriminator(ganDInput_dict["df" + str(self.train_df) + "_" + lossName]))
                            elif lossName in ["portW", "per", "utility"]:
                                G_loss -= self.loss_weight_dict[lossName] * torch.mean(discriminator(ganDInput_dict["df" + str(self.train_df) + "_rg" + str(self.train_risk_gamma) + "_" + lossName]))

                    #backward generator gradients update
                    G_loss.backward()#retain_graph = True)
                    self.generator_solver.step()

		    # Housekeeping - reset gradient
                    self.generator.zero_grad()
                    if self.isStacking:
                        self.discriminator.zero_grad()
                    else:
                        for lossName, discriminator in self.discriminator_dict.items():
                            discriminator.zero_grad()

            #eval step
            if epoch % self.evalFreq == 0:
                z = Variable(torch.randn(self.eval_num_samples * (self.strat_train_days + self.strat_val_days), self.z_dim).cuda())           
                G_data = self.generator(z)
                G_data = G_data.reshape((self.eval_num_samples, self.strat_train_days + self.strat_val_days, self.mu_dim))

                df_data_dict = OrderedDict()
                df_data_dict[self.train_df] = G_data
                ganDInput_dict = self.computeDInput(df_data_dict, self.eval_risk_gamma_list, self.D_input_dim_eval_dict, "eval")[0]#this one is tricky, since we are gonna to evaluate on training set for different gan loss

                if self.isPrintOpen:
                    print("Iter: ", epoch)
                    print("D_loss: ", D_loss.data.cpu().numpy(), "G_loss: ", G_loss.data.cpu().numpy())
                
                if self.isStacking:
                    csv.writer(self.output_fakeWorld_file).writerow([epoch, D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy()] + self.computeStats([ganDInput_dict]) + self.computeWass([self.train_D_input_dict], ganDInput_dict) + self.computeWass(self.eval_D_input_dict_list, ganDInput_dict))
                else:    
                    csv.writer(self.output_fakeWorld_file).writerow([epoch, D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy()] + self.computeDGLoss(batchDInput_dict, ganDInput_dict) + self.computeStats([ganDInput_dict]) + self.computeWass([self.train_D_input_dict], ganDInput_dict) + self.computeWass(self.eval_D_input_dict_list, ganDInput_dict))

        torch.save({"generator_state_dict": self.generator.state_dict()}, self.output_generator_file)       
        self.output_fakeWorld_file.close()

   
def multivariate_t(m, S, df=np.inf, n=1):
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

if __name__=='__main__':
    import dill
    with open("config.json") as json_file:
        config = json.load(json_file)
        true_mu = np.array(config["true_mu"])
        true_cov = np.array(config["true_cov"])
        train_df = config["train_df"]
        eval_df_list = config["eval_df_list"]
        true_beta = config["true_beta"]
        z_dim = config["z_dim"] 
        h_dim = config["h_dim"]       
        isStacking = config["isStacking"]
        loss_weight_dict = config["loss_weight_dict"]
        warm_up_samples = config["warm_up_samples"]
        train_num_samples = config["train_num_samples"]
        eval_num_samples = config["eval_num_samples"]
        batch_size = config["batch_size"]
        strat_train_days = config["strat_train_days"]
        strat_val_days = config["strat_val_days"]
        n_epoch = config["n_epoch"]
        n_critics = config["n_critics"]
        n_actors = config["n_actors"]
        lr = config["lr"]
        clip_lb = config["clip_lb"]
        clip_ub = config["clip_ub"]
        shrink = config["shrink"]
        isPrintOpen = config["isPrintOpen"]
        evalFreq = config["evalFreq"]
        isUsingNewData = config["isUsingNewData"]
        train_risk_gamma = config["train_risk_gamma"]
        eval_risk_gamma_list = config["eval_risk_gamma_list"]
        isRealData = config["isRealData"]
        isParaGen = config["isParaGen"]
        sourceEtfNameList = config["sourceEtfNameList"]
        halfLifeList = config["halfLifeList"]
        estHalfLife = config["estHalfLife"]
        targetEtfNameList = config["targetEtfNameList"]
        trainHorizonRange = config["trainHorizonRange"]
        evalHorizonRange = config["evalHorizonRange"]
        horizonDecay = config["horizonDecay"]
        hasAutoRegBias = config["hasAutoRegBias"]
        hasFeatureInLoss = config["hasFeatureInLoss"]

    print("true mu: ", true_mu, "true cov: ", true_cov)

    dateToday = date.today().strftime("%Y%m%d")
    subFolderID = 0
    while os.path.exists('out/' + dateToday + '/' + str(subFolderID) + "/"):
        subFolderID += 1
    os.makedirs('out/' + dateToday + '/' + str(subFolderID) + "/")
    
    # Train the model
    datGAN = DAT_GAN(true_mu, true_cov, train_df, eval_df_list, true_beta, z_dim, h_dim, isStacking, loss_weight_dict, warm_up_samples, train_num_samples, eval_num_samples, batch_size, strat_train_days, strat_val_days, n_epoch, n_critics, n_actors, lr, clip_lb, clip_ub, shrink, isPrintOpen, evalFreq, isUsingNewData, train_risk_gamma, eval_risk_gamma_list, dateToday, subFolderID, isRealData, isParaGen, sourceEtfNameList, halfLifeList, estHalfLife, targetEtfNameList, trainHorizonRange, evalHorizonRange, horizonDecay, hasAutoRegBias, hasFeatureInLoss)
    if isRealData:
        datGAN.train_CGAN_RT()
    else:
        datGAN.trainGAN()
