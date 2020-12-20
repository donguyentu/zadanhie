import numpy as np
import matplotlib.pyplot as plt
from math import pi

f = open("C:/results/test.txt", "a")
x = -5.12
def y(x): 
  y = 10 + x**2 - 10*np.cos(2*pi*x)
  y1 = np.round(y,2)
  return y1


f.write("X         Y")

while x <= 5.12: 
 f.write("\n" + str(x) + "         " + str(y(x)))
 x = x + 0.1

f.close()

x = np.arange(-5.12, 5.12)
fig, ax = plt.subplots()
ax.plot(x, y(x))  
lgnd = ax.legend(['y'], loc='upper center', shadow=True)
lgnd.get_frame().set_facecolor('green')
plt.show()
