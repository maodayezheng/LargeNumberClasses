import json

batch = []
with open('ProcessedData/frequency.txt', 'r') as data:
        d = json.loads(data.read())
        print(len(d))