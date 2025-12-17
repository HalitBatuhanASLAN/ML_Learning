# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:06:44 2025

@author: halit
"""

#%%numpy library

import numpy as np

arr1 = np.array([1,2,3,4,5])

vector = np.array([1,2,3,4,5,6,7,8,9,0])

matrix = vector.reshape(2,5)

print(matrix.shape)
print(matrix.ndim)
print(matrix.dtype)
print(matrix.size)


#özel matrisler

zeroMatris = np.zeros((3,5))

rastgeleMatris = np.random.rand(4,5)

linMatris = np.linspace(10,200,5)


#dört işlem

a = np.array([10,20,30])
b = np.array([1,2,3])

toplam = a + b

transposeMatrix = matrix.T

maxElement = rastgeleMatris.max()
print(maxElement)

print("Matrix mean is", rastgeleMatris.mean())


print(rastgeleMatris[1,3])
print(rastgeleMatris[1:])
print(matrix[1:2,0:4])



v1 = np.array([1,2])
v2 = np.array([3,4])

yatayVektor = np.hstack((v1,v2))
dikeyVektor = np.vstack((v2,v1))

d = np.array([1,2,3])

e = d
e[0] = 50

# vektör türetme
e = d.copy()








