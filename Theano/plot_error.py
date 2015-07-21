#coding: utf-8
import matplotlib.pyplot as plt

validation_epoch = []
validation_error = []

fp = open("validation_error.txt")
for line in fp:
    if line == "": continue
    line = line.strip().split()
    validation_epoch.append(int(line[0]))
    validation_error.append(float(line[1]))
fp.close()

plt.plot(validation_epoch, validation_error, "r-")
plt.xlabel("epoch")
plt.ylabel("validation error (%)")
plt.grid()
plt.tight_layout()
plt.show()
