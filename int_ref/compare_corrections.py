import matplotlib.pyplot as plt
import numpy as np

print 'Let\'s consider the difference between the additive correction and the correct correction'
a=-8.099529E-6
b=706378.392880E-6
ap = b
cp = -a/ap
temp = np.linspace(0,40,1000)
print temp
z = [a*t+b for t in temp]
zp = [ap*(1/(1 + cp*t)) for t in temp]
plt.plot(temp,z,color='blue')
plt.plot(temp,zp,color='green')
plt.show()


zd = [z1-z2 for z1,z2 in zip(z,zp)]
plt.plot(temp,zd)
plt.show()
