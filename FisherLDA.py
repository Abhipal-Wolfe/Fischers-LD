import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import sqrt, log


name_column = {"dataset_FLD.csv": ["f0","f1","f2","Target"]}

flda = pd.read_csv("dataset_FLD.csv", header=None, names=name_column["dataset_FLD.csv"], sep=",")

flda_positive, flda_negative = [ x for _, x in flda.groupby(flda["Target"] == 0)]

pos_class = np.array(flda_positive.drop("Target", 1))
neg_class = np.array(flda_negative.drop("Target", 1))

fig=plt.figure()
axis=plt.axes(projection='3d')
pos_f0=np.array(flda_positive["f0"])
pos_f1=np.array(flda_positive["f1"])
pos_f2=np.array(flda_positive["f2"])
axis.scatter(pos_f0,pos_f1,pos_f2,color="blue")
neg_f0=np.array(flda_negative["f0"])
neg_f1=np.array(flda_negative["f1"])
neg_f2=np.array(flda_negative["f2"])
axis.scatter(neg_f0,neg_f1,neg_f2,color="red")
axis.set_title("Points in original space. Blue-pos Red-neg")
plt.show()


m1 = np.mean(pos_class, axis=0)
m2 = np.mean(neg_class, axis=0)

sw1 = np.dot((pos_class-m1).T, (pos_class-m1))
sw1/=pos_class.shape[0]

sw2 = np.dot((neg_class-m2).T, (neg_class-m2))
sw2/=neg_class.shape[0]

sw=sw1+sw2
        
print("Sw: ")
print(sw)

W = np.dot(np.linalg.inv(sw), (m1-m2).T)
print("w: ")
print(W)


projected_posclass = np.sort(np.dot(pos_class, W.T))
projected_negclass = np.sort(np.dot(neg_class, W.T))

fig, axis = plt.subplots()
axis.plot(projected_posclass, np.zeros(len(projected_posclass)), color="blue", marker="+")
axis.plot(projected_negclass, np.zeros(len(projected_negclass)), color="red", marker="+")
axis.set_title("Points after being projected in 1D. Blue-pos Red-neg")
plt.show()


m1 = np.mean(projected_posclass)
m2 = np.mean(projected_negclass)
v1 = np.var(projected_posclass)
v2 = np.var(projected_negclass)
a = (1/v2 - 1/v1)
b = 2*(m1/v1 - m2/v2)
c = (((m2**2)/v2 - (m1**2)/v1) + log(v2/v1))
first_sol = (-b + sqrt(((b**2) - 4*a*c)))/(2*a)
second_sol = (-b - sqrt(((b**2) - 4*a*c)))/(2*a)

print("Threshold value: ")
print(first_sol)


pos_normal_curve = (1/sqrt(2*np.pi*v1)) * np.exp((-((projected_posclass-m1)**2)/(2*v1)))
neg_normal_curve = (1/sqrt(2*np.pi*v2)) * np.exp((-((projected_negclass-m2)**2)/(2*v2)))

x = np.array([first_sol]*3)
y = np.linspace(0, 0.3, 3)
fig, axis = plt.subplots()
axis.plot(projected_posclass, pos_normal_curve, color="blue")
axis.plot(projected_negclass, neg_normal_curve, color="red")
axis.plot(x, y, color="black")
axis.set_title("Normal distributions. Blue-pos Red-neg")
plt.show()


print("Equation of unit vector in 3D")
print(str(W[0])+"*x+"+str(W[1])+"*y+"+str(W[2])+"*z="+str(first_sol))





















