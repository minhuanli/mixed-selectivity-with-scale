import numpy as np
from scipy.stats import norm
from scipy import integrate


#=========== Related Functions ============

def function_H(T):
    return 1-norm.cdf(T)

def function_in_G(h,x,T):
    
    return np.exp(-h**2/2)*function_H( ((1-x)*h-T) / np.sqrt(x*(2-x)) )/np.sqrt(2*np.pi)

def function_G(delta_s,f,T):
    
    Int_res, err = integrate.quad(function_in_G,T,np.inf,args=(delta_s,T))
    
    return Int_res/(f*(1-f))


def q1_func(f,T,delta_sigma):
    
    return f-f*(1-f)*function_G(delta_sigma,f,T)

def q2_func(f,T,delta_eta):
    
    return f-f*(1-f)*function_G(delta_eta,f,T)


def q_func(p1,p2,p3,f,T,delta_sigma,delta_eta):
    
    delta_mix = (delta_sigma + delta_eta) / 2.
    
    delta_m = p1*function_G(delta_sigma,f,T) + p2*function_G(delta_eta,f,T) +  p3*function_G(delta_mix,f,T)
    
    return f - delta_m*f*(1-f)

def numerator_func(Nc,q,f):
    return Nc**2 * (q-f**2)**2


def part1_func(Nc,q,f):
    
    return Nc* ( q*(1-2*f)**2 + 2*f**3 - 3*f**4 )


def part3_func(Nc,P,K,p1,p2,f,q1,q2):
    
    uni_term = Nc*(P*K-1) * f**2 * (1-f)**2
    
    excess_term = Nc*(2*f-1)**2 * ( (K-1)*p1*(q1-f**2) + (P-1)*p2*(q2-f**2) ) 
    
    return uni_term + excess_term

def part4_func(Nc,N,M,P,K,p1,p2,p3,f,T,q1,q2):
    
    term1 = (Nc*p1)**2 * (K-1) * (q1-f**2)**2
    term2 = (Nc*p2)**2 * (P-1) * (q2-f**2)**2
    term3 = 2*Nc**2*p1*p2*(K-1)*(P-1)*(q1-f**2)*(q2-f**2) / (P*K-1)
    term4 = Nc**2 * (P*K-K) * p1**2 * (1/N) *np.exp(-2*T**2)/(2*np.pi)**2
    term5 = Nc**2 * (P*K-P) * p2**2 * (1/M) *np.exp(-2*T**2)/(2*np.pi)**2
    term6 = Nc**2 * (P*K-1) * p3**2 * (1/(N+M)) *np.exp(-2*T**2)/(2*np.pi)**2
    
    return term1 + term2 + term3 + term4 + term5 + term6
    
def Readerror_theory(Nc,N,M,P,K,p1,p2,p3,f,delta_sigma,delta_eta):
    
    T = norm.ppf(1-f)
    q1 = q1_func(f,T,delta_sigma)
    q2 = q2_func(f,T,delta_eta)
    q = q_func(p1,p2,p3,f,T,delta_sigma,delta_eta)
    
    numerator = numerator_func(Nc,q,f)
    part1 = part1_func(Nc,q,f)
    part3 = part3_func(Nc,P,K,p1,p2,f,q1,q2)
    part4 = part4_func(Nc,N,M,P,K,p1,p2,p3,f,T,q1,q2)
    
    SNR = numerator/ (part1 + part3 + part4)
    
    error = function_H(np.sqrt(SNR))
    
    return error,SNR


#===========Example Codes to generate theory curve================
N = 100
M = 100
Nc = 1000

P = 40
K = 5
delta_sigma = 0.1
delta_eta_list = [0.1,0.3,0.5]
p1 = 0.1
p2 = 0.1
p3 = 0.8

flist = 10**np.linspace(-2,-0.4,20)

ErrRecord_f_theory_1 = np.zeros((3,100,2))
flist_dense = 10**np.linspace(-2.1,-0.4,100)

for j,f in enumerate(tqdm_notebook(flist_dense)):
    
    for i,delta_eta in enumerate(delta_eta_list):
        
        ErrRecord_f_theory[i,j,0],ErrRecord_f_theory[i,j,1] = Readerror_theory(Nc,N,M,P,K,p1,p2,p3,f,delta_sigma,delta_eta)















