import pymysql 
import numpy as np
import pandas as pd
import math
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import kendalltau
from scipy.special import gamma
from statsmodels.distributions.empirical_distribution import ECDF
from pynverse import inversefunc as inv
import sys

class copula:
    
    prs = dict()
    #corr_mat = pandas.DataFrame()
    instruments = []
    
    def __init__(self, prs):
        prs = prs
        return(None)
    
    def price_return(self, dataset, i):
        returns = np.log(dataset[self.instruments[i]]/dataset[self.instruments[i]].shift(1))
        return returns 
                               
    def copula_params(self, family, dataset):
        tau=kendalltau(x=dataset[self.instruments[0]],y=dataset[self.instruments[1]])[0]
        rho=np.sin(np.pi/2*tau)
        if  family == 'clayton':
            return 2*tau/float(1-tau)
        elif family == 'frank':
            integrand = lambda t: t/(np.exp(t)-1)
            frank_fun = lambda theta: ((tau - 1)/4.0  - (quad(integrand, sys.float_info.epsilon, theta)[0]/theta - 1)/theta)**2
            return minimize(frank_fun, 4, method='BFGS', tol=1e-5).x
        elif family == 'gumbel':
            return 1/(1-tau)     
        elif family == 'student-t':            
            return rho

    def log_pdf_copula(self, family, dataset, student_df=None):
        theta=self.copula_params(family,dataset)
        returns_0 = self.price_return(dataset, 0).dropna() #x
        returns_1 = self.price_return(dataset, 1).dropna() #y
        x= ECDF(returns_0)(returns_0)
        y= ECDF(returns_1)(returns_1)
        pdf_list=[]
        if  family == 'clayton':
            for (u,v) in zip(x,y):
                pdf = (theta+1) * ((u**(-theta)+v**(-theta)-1)**(-2-1/theta)) * (u**(-theta-1)*v**(-theta-1))
                pdf_list.append(pdf)           
        elif family == 'frank':
            for (u,v) in zip(x,y):
                num = -theta *(np.exp(-theta)-1) * (np.exp(-theta*(u+v)))
                denom = ((np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta)-1))**2
                pdf = num/denom
                pdf_list.append(pdf)
        elif family == 'gumbel':
            for (u,v) in zip(x,y):
                A = (-np.log(u))**theta + (-np.log(v))**theta
                #C = np.exp(-A**(1/theta))
                C = np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))
                pdf = C * (u*v)**(-1) * (A**(-2+2/theta))*((np.log(u)*np.log(v))**(theta-1))*(1+(theta-1)*A**(-1/theta))
                pdf_list.append(pdf)
        elif family=='student-t':
            for (u,v) in zip(x,y):
                rho = theta
                n = student_df
                pdf_x   =  lambda x: gamma((n+1)/2)/(np.sqrt(n*np.pi)*gamma(n/2))*(1+x**2/n)**(-(n+1)/2) #pdf of x
                #pdf_xy =  lambda x,y: 1/(gamma(n/2))*gamma((n+2)/2)/(n*np.pi) *  (1/np.sqrt(rho))* (1 + (x**2)/(n*rho)+(y**2)/(n*rho))**(-((n+2)/2))     #joint pdf of x,y
                tn      =  lambda h: quad(pdf_x, -math.inf, h)[0]     #CDF of x
                #t2n    =  lambda h,k : dblquad (pdf_xy, -math.inf,h, lambda x: -math.inf,lambda x:k)[0]        #CDF of x,y          
                pdf     =   1/np.sqrt(1-rho**2) * gamma((n+2)/2)*gamma((n/2))/(gamma((n+1)/2)**2) * ((1+inv(tn)(u)**2/n)*(1+inv(tn)(v)**2/n))**((n+1)/2) / (1+1/(n*(1-rho**2))*(inv(tn)(u)**2-2*rho*inv(tn)(u)*inv(tn)(v)+inv(tn)(v)**2))**((n+2)/2)            
                pdf_list.append(pdf)
        return np.log(pdf_list)
    
    def opt_aic(self, dataset):
        family=['clayton','frank','gumbel']
        AIC_values={'clayton':[],'frank':[], 'gumbel':[]}
        student_aic={'t_5':[],'t_10':[],'t_15':[],'t_20':[],'t_25':[],'t_30':[]} 
        for i in family:
            if i == 'student-t':
                for j in [5,10,15,20,25,30]:
                    log_pdf = self.log_pdf_copula(family=i,dataset=dataset,student_df=j)
                    loglikehood=sum(np.nan_to_num(log_pdf))
                    student_aic['t_'+str(j)]=-2*loglikehood+2
                student_aic=pd.DataFrame.from_dict(student_aic,orient='index')
                student_min=student_aic.min(axis=0)[0]
                student_df=float(student_aic[student_aic==student_min].dropna().index[0].split('_')[0])
                AIC_values[i]=student_min
            else:
                log_pdf = self.log_pdf_copula(family=i,dataset=dataset)
                loglikehood=sum(np.nan_to_num(log_pdf))
                AIC_values[i]=-2*loglikehood+2
        AIC_values=pd.DataFrame.from_dict(AIC_values,orient='index')
        AIC_min=AIC_values.min(axis=0)[0]
        copula_type=AIC_values[AIC_values==AIC_min].dropna().index[0]
        return copula_type
    
    def Misprice_Index(self, dataset, student_df=None):  
        family =self.opt_aic(dataset)  
        #family= 'student-t'
        theta=self.copula_params(family,dataset)
        returns_0 = self.price_return(dataset,0) #x
        returns_1 = self.price_return(dataset,1) #y
        u=ECDF(returns_0)(returns_0.tail(1))
        v=ECDF(returns_1)(returns_1.tail(1))
        MI_0=None
        MI_1=None
        if  family == 'clayton':
            MI_0=v**(-theta-1)*(u**(-theta)+v**(-theta)-1)**(-(1/theta)-1)
            MI_1=u**(-theta-1)*(u**(-theta)+v**(-theta)-1)**(-(1/theta)-1)        
        elif family == 'frank':
            MI_0=((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta*u)-1))/ \
                 ((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta)-1))           
            MI_1=((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta*v)-1))/ \
                ((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta)-1))               
        elif family == 'gumbel':
            A  = (-np.log(u))**theta + (-np.log(v))**theta
            #C = np.exp(-A**(1/theta)) #Gumbel copula
            C    = np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))
            MI_0 = C*(A**((1-theta)/theta))*(-np.log(v))**(theta-1)*(1/v)
            MI_1 = C*(A**((1-theta)/theta))*(-np.log(u))**(theta-1)*(1/u)    
        elif family== 'student-t':
            rho=theta
            n = student_df
            pdf_x     = lambda x: gamma((n+1)/2)/(np.sqrt(n*np.pi)*gamma(n/2))*(1+x**2/n)**(-(n+1)/2) #pdf of x
            pdf_x_2   = lambda x: gamma((n+1+1)/2)/(np.sqrt((n+1)*np.pi)*gamma((n+1)/2))*(1+x**2/(n+1))**(-(n+1+1)/2) #pdf of x with degree n+1
            #pdf_xy   = lambda x,y: 1/(gamma(n/2))*gamma((n+2)/2)/(n*np.pi) *  (1/np.sqrt(rho))* (1 + (x**2)/(n*rho)+(y**2)/(n*rho))**(-((n+2)/2))     #joint pdf of x,y
            tn        = lambda h: quad(pdf_x, -math.inf, h)[0]     #CDF of x
            tn_2      = lambda h: quad(pdf_x_2, -math.inf, h)[0]     #CDF of x with degree n+1
            #t2n       = lambda h,k : dblquad (pdf_xy, -math.inf,h, lambda x: -math.inf,lambda x:k)[0] #CDF of x,y         
            MI_0      = tn_2(np.sqrt((n+1)/(n+inv(tn)(v)**2))*(inv(tn)(u)-rho*inv(tn)(v))/np.sqrt(1-rho**2))
            MI_1      = tn_2(np.sqrt((n+1)/(n+inv(tn)(u)**2))*(inv(tn)(v)-rho*inv(tn)(u))/np.sqrt(1-rho**2))
        misc = [MI_0, MI_1]
        MI_0 = misc[0].tolist()[0]
        MI_1 = misc[1].tolist()[0]
        return [MI_0, MI_1]