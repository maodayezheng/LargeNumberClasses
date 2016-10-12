import matplotlib.pyplot as plt
import numpy as np
import json
import glob, os


alpha = ["0.0", "0.75", "1.0"]

for a in alpha:
    fig = plt.figure()
    fig.canvas.set_window_title('a ='+a)
    labels=[]
    for file in glob.glob("results/*_"+a+"*.txt"):
        r = np.loadtxt(file)
        plt.plot(r)
        names = file.split("_")
        name = names[1]
        labels.append(name)
    plt.legend(labels)
    plt.xlabel("iterations (10^3)")
    plt.ylabel("exact log likelihood")
    plt.title("Training on 100,000 sentences with distortion factor ("+a+")")
    plt.show()

with open("ProcessedData/frequency.txt", "r") as f:
        d = json.loads(f.read())
        print(len(d))

