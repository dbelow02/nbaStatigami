#Class definition for PDF
#PDF abstracts away the loading, fitting, predicting, and saving of PDF functions necessary for extracting
#probabilities from a given feature
#A Specific feature in the dataset may have one kind of PDF, ie binomial, while another may be completely different.

# also defines a second class (PDFCollector) obscuring the handling of multiple PDF objects into a dict-like structure.





#Reasoning for using a PDF to generate percentile values for the data is not necessarily to give us better data.
#In fact, using empirical percentile data against the exact data (1.6M rows) would give more accurate percentiles.
#However, this approach makes compute dramatically faster as we don't need 1.6M rows compute each time, it's baked into PDF
#It also handles extrapolation.. What if someone scores 400 points?
#Also, it helps to unify the "score" calculation process. In some cases, z score is better for columns, others percentile.
#For example, blocks is better as percentile (low standard deviation, low mean = high z score from little change)
#Whereas z score is better for points, it determines more accurately HOW good or HOW bad.
#The method of using a pdf, then "converting" to z score bakes these two benefits into one action, without extra complication.



import pandas as pd
import pickle
import os
from scipy.optimize import minimize
from scipy import stats
import numpy as np

class PDF:

    def __init__(self, featureName, newBool, data=pd.DataFrame, pdfType=None, saveDir=None, fileName=None):
        if not newBool:
            if saveDir == None or fileName == None:
                raise ValueError("saveDir and fileName are required when newBool = False")
            path = os.path.join(saveDir, fileName)
            self.loadExisting(path)
            # The pickle stores the folder it was saved from; use the current one instead
            self.saveDir = saveDir
            self.fileName = fileName
        else:
            self.name = featureName
            self.data = data
            self.pdfType = pdfType
            self.model = None
            self.saveDir = saveDir
            self.fileName = fileName
        return


    def fit(self):
       
        if self.pdfType == None or self.data.empty:
            print("pdfType or data not initialized for PDF object. Use setPDFType / setData to set them")
            print("Typically, for discrete stats like blocks and steals, use negative binomial or poisson")
            print("For roughly normal continous percentages, use beta (note that 0's will be trimmed)")
            print("For roughly normal, wide range like points, minutes - use normal or gamma")

        print(f"Fitting PDF for feature {self.name} of type {self.pdfType}")
        if self.pdfType == "nbinom":
            self.fit_negative_binomial()
        elif self.pdfType == "beta":
            self.fit_beta()
        elif self.pdfType == "poisson":
            self.fit_poisson()
        elif self.pdfType == "normal":
            data = self.data.dropna()
            self.model = stats.norm.fit(data) #Returns (loc, scale)
        elif self.pdfType == "gamma":
            data = self.data.dropna()
            self.model = stats.gamma.fit(data) #Returns (a, loc, scale)
        else:
            print(f"{self.pdfType} not supported, please use nbinom, beta, poisson, normal, or gamma")
            return
        #Uses scipy to fit specified model to data provided. Stores model in self.model (not sure about data type yet)
        self.data = None #Don't need to keep the data once fit.
    def fit_negative_binomial(self):
        data = self.data.dropna().astype(int)
    
        def neg_log_likelihood(params):
            n, p = params
            if n <= 0 or p <= 0 or p >= 1:
                return float("inf") #This guess is infinitely bad, look somewhere else (not valid parameters for negative binomial)
            return -np.sum(stats.nbinom.logpmf(data, n, p))
        
        #method of moments
        mean = data.mean()
        var = data.var()
        p0 = mean / var
        n0 = mean * p0 / (1 - p0)
        
        result = minimize(neg_log_likelihood, [n0, p0], method="Nelder-Mead")
        #print(result.x)
        #Should be (n, p)
        self.model = result.x
        return
        #Manual MLE as scipy doesn't support fit very well for this

    def fit_beta(self):
        data = self.data.dropna()
        #Beta expects (0,1), so need to trim to be just above and below 0/1
        epsilon = 1e-6
        data = data.clip(lower=epsilon, upper=1 - epsilon)
        self.model = stats.beta.fit(data, floc=0, fscale=1) 
        #fix bounds to [0,1]
        #returns in format (a, b, loc, scale)
        return
    
    def fit_poisson(self):
        data = self.data.dropna().astype(int)
        #MLE for poisson is just the mean
        lambda_hat = data.mean()
        self.model = (lambda_hat,)
        return

    def save(self):
        path = os.path.join(self.saveDir, self.fileName)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f)
        os.replace(tmp, path)   # the old file stays intact until the new one is complete (atomicity)
        print(f"Saved {self.name} PDF object to {path}")
        
    def getProbability(self, observation):
        #Uses self.model to find the probability of an observation, returns probability as double
        #Uses something like stats.norm.cdf(value, *params) for each
        percentile = None
        if self.pdfType == "nbinom":
            percentile = stats.nbinom.cdf(observation, *self.model)
        elif self.pdfType == "beta":
            if observation < 0 or observation > 1:
                print("Observation should be a percentage for model type beta.")
                return np.nan
            percentile = stats.beta.cdf(observation, *self.model)
        elif self.pdfType == "gamma":
            percentile = stats.gamma.cdf(observation, *self.model)
        elif self.pdfType == "poisson":
            percentile = stats.poisson.cdf(observation, *self.model)
        elif self.pdfType == "normal":
            percentile = stats.norm.cdf(observation, *self.model)
        else:
            print(f"{self.pdfType} not found in allowed list")
            return None

        if percentile == None:
            print("Issue getting percentile.")
            return None
        
        return percentile

    def setPDFType(self, pdfType):
        if pdfType not in ["nbinom", "poisson", "beta", "normal", "gamma"]:
            print(f"{self.pdfType} not supported, please use nbinom, beta, poisson, normal, or gamma")
            return
        self.pdfType = pdfType
        return

    def setData(self, data: pd.DataFrame):
        if data.empty:
            print("Data provided is empty")
            return
        self.data = data
        return

    def loadExisting(self, filepath):
        with open(filepath, "rb") as f:
            instance = pickle.load(f)
        self.__dict__.update(instance.__dict__)
        print(f"Loaded {self.name} PDF object from {filepath}")
        return


class PDFCollector:
    #Collection of PDF types.
    #Essentially a dict wrapper.
    #Want to be able to do PDFCollector[featureName]
    #Should instantiate / load all feature PDF's
    def __init__(self):
        self.featurePDFs = {}
        return
    
    def load(self, pdfBoolDict):
        #Takes dict of feature -> (newBool, filepath, filename)
        for feature in pdfBoolDict.keys():
            self.featurePDFs[feature] = PDF(feature, newBool=pdfBoolDict[feature][0], saveDir=pdfBoolDict[feature][1], fileName=pdfBoolDict[feature][2])
        return
    
    def save(self, featureName=None):
        if featureName is None:
            for feature in self.featurePDFs.keys():
                self.featurePDFs[feature].save()
        else:
            self.featurePDFs[featureName].save()

    def __getitem__(self, feature):
        return self.featurePDFs[feature]

    def __setitem__(self, feature, value):
        self.featurePDFs[feature] = value
        return


if __name__ == "__main__":
    print("pf.py should not be called directly")
