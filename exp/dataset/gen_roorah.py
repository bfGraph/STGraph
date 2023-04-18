import numpy as np
from rich import inspect

roorah_file = 'roorah/'

edgeSrc = np.array([0,2,3,4])
edgeDst = np.array([1,1,1,1])
edgeNorm = np.array([0.3,0.2,0.4,0.5])
edgeType = np.array([1,1,2,2])
labels = np.array([0,1,0,0,0])
trainIdx = np.array([0,1,2,3,4])
testIdx = np.array([0,1,2,3,4])

np.save(roorah_file + 'edgeSrc.npy', edgeSrc)
np.save(roorah_file + 'edgeDst.npy', edgeDst)
np.save(roorah_file + 'edgeNorm.npy', edgeNorm)
np.save(roorah_file + 'edgeType.npy', edgeType)
np.save(roorah_file + 'labels.npy', labels)
np.save(roorah_file + 'trainIdx.npy', trainIdx)
np.save(roorah_file + 'testIdx.npy', testIdx)

f = open(roorah_file + "num.txt", 'w')
f.write("5#2#2")
f.close()