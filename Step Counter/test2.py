import scipy
import matplotlib

mat = scipy.io.loadmat("sensorlog_20241114_143501.mat")
print((mat.keys()))
acc = mat.get("__function_workspace__")
print(acc)
print(len(acc))