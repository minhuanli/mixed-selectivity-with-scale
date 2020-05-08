import numpy as np


def cal_deltaC(matrix_C,f=0.5):
    res = 0.
    Nr,Nc = matrix_C.shape
    center = matrix_C[0,:]
    
    for i in range(1,Nr):
        
        res = res + np.abs(matrix_C[i,:]-center).sum()
    
    return res/(2*Nc*f*(1-f)*(Nr-1))



def cal_excessQ(matrix_C,f,Ns):
    
    Nr,Nc = matrix_C.shape
    r_mn = []
    
    for m in range(Nr):
        for n in range(m+1,Nr):
            temp_rmn = (matrix_C[m,:]*matrix_C[n,:]).mean() - f*f
            r_mn.append(temp_rmn)
    r_mn = np.array(r_mn)
    r_mn_square = r_mn**2
    Q = np.sqrt((r_mn_square.mean() / (f**2*(1-f)**2) - 1./float(Nc))*Ns)
    dev_Q = np.sqrt((np.std(r_mn_square)/ (f**2*(1-f)**2) - 1./float(Nc))*Ns)
    
    return Q, dev_Q