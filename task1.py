import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

x = -5.12
def y(x): 
  y = 10 + x**2 - 10*np.cos(2*pi*x)
  y1 = np.round(y,2)
  return y1

# создание файла
try:
  os.mkdir('results')
except OSError:
  pass
complete_file = os.path.join('results', 'task_01_307B_do_1.txt')
f = open(complete_file, 'w')
# текстовый файл с таблицей
f.write("X         Y")

while x <= 5.12: 
 f.write("\n" + str(x) + "         " + str(y(x)))
 x = x + 0.1

f.close()

x = np.arange(-5.12, 5.12,0.01)
fig, ax = plt.subplots()
ax.plot(x, y(x))  
lgnd = ax.legend(['y'], loc='upper center', shadow=True)
lgnd.get_frame().set_facecolor('green')
plt.show()
