import numpy as np
import matplotlib.pyplot as plt

#EX2-3.
eleNum1 = [60,64,70,79,92,104,127,162,224,369,1100,1223,1572,2197,3660,10970,12188,15673,21936,36557,109663,121848,156661]

maxDis1 = [0.13576 for x in range(len(eleNum1))]

maxStress1 = [6125.1,6125.1,6255.3,6356.6,6503.9,6559.2,6680.7,6783,6896.8,7001.1,7111.6,7117.2,
              7128.3,7139.2,7150.3,7161.3,7161.9,7163,7164.1,7165.2,7166.3,7166.3,7166.4]


plt.plot(eleNum1,maxStress1)
plt.grid()
plt.xlabel("The number of elements")
plt.ylabel("Maximum stress (psi)")
plt.show()

plt.plot(eleNum1,maxDis1)
plt.grid()
plt.xlabel("The number of elements")
plt.ylabel("Maximum displacement (in)")
plt.show()

#EX2-4
thick = np.array([0.4,0.39,0.35,0.34,0.339,0.335])-0.25
dt = []
piv = thick[0]
for ele in thick:
    dt.append(piv-ele)
stress = [7166.4,7885.3,12271,13995,14118,15011]
cr = [14000 for x in range(len(stress))]
plt.plot(dt,stress,label="Stress")
plt.plot(dt,cr,'--',label="Stress is 14000psi")
plt.xlabel("amount of reduced thickness")
plt.ylabel("Maximum stress (psi)")
plt.legend()
plt.grid()
plt.show()


#EX2-5
eleNum2 = [60,64,70,79,92,104,127,162,224,369,1100,1223,1572,2197,3660,10970,12188,15673,21936,36557,109663,121848,156661]

maxDis2 = [0.39316 for x in range(len(eleNum2))]

maxStress2 = [17701,17701,18078,18370,18796,18965,19307,19603,19932,20233,20552,20569,20601,20632,
              20664,20696,20698,20701,20704,20707,20710,20711,20711]

plt.plot(eleNum2,maxStress2)
plt.xlabel("The number of elements")
plt.ylabel("Maximum stress (psi)")
plt.grid()
plt.show()

plt.plot(eleNum2,maxDis2)
plt.grid()
plt.xlabel("The number of elements")
plt.ylabel("Maximum displacement (in)")
plt.show()

