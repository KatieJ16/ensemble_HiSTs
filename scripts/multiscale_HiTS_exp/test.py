from datetime import datetime
import time
import numpy

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

for i in range(10):
    numpy.save(str(i) + '.npy', range(1,100))
    time.sleep(5.0)
