import numpy as np

#构造张量
A = np.arange(4)
B = np.arange(4*5).reshape(4,5)
C = np.arange(4*5*6).reshape(4,5,6)
print("A=",A,";","B=",B,";","C=",C)


#1.张量积 AB
AB = np.einsum('i,jk->ijk',A,B)


#2.张量B的第二个轴与张量C的第二个轴缩并
D = np.einsum('ij,kjm->ikm',B,C)


#3.张量D与张量A的点积
DA = np.einsum('ikm,k->im',D,A)


#4.张量B与其转置的点积
BBT = B @ B.T


#验证
print("AB=",AB,";","AB.shape:",AB.shape)
print("D=",D,";","D.shape:",D.shape)
print("DA=",DA,";","DA.shape:",DA.shape)
print("BBT=",BBT,";","BBT.shape:",BBT.shape)