import json
import numpy as np

batch = []
sentences = [0] * 1000
with open('ProcessedData/sentences.txt', 'r') as data:
        for d in data:
            s = json.loads(d)
            sentences[len(s)] += 1
        data.close()

np.savetxt("ProcessedData/sent_his.txt", sentences)

