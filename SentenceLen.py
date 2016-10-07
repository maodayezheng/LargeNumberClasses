import json

batch = []
with open('ProcessedData/sentences.txt', 'r') as data:
    for d in data:
        d = json.loads(d)
        if len(d) > 70:
            continue
        batch.append(d)
    data.close()
    print(len(batch))