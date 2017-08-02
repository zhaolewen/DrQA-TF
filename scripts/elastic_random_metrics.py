import time, requests, random
import math

step = 0
step_per_epoch = 25
stepCount = 1000

elasticData = {"name":"test_nn"}

elasticData["launch_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

for step in range(stepCount):
    elasticData['phase'] = "train"
    elasticData['step'] = step
    elasticData['epoch'] = int(step/step_per_epoch)
    elasticData['step_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

    while True:
        precision = 1.0-math.exp(-step*0.01)*random.uniform(0,1)
        if precision<1.0:
            break
    while True:
        recall = 1.0-math.exp(-step*0.01)*random.uniform(0,1)
        if recall<1.0:
            break

    elasticData["precision"] = precision
    elasticData["recall"] = recall
    elasticData["learning_rate"] = 0.001 * math.exp(-step*0.001)
    elasticData["loss"] = 100*math.exp(-step)

    elasticData["f1"] = 2.0* (precision * recall)/(precision + recall)

    r = requests.post("http://localhost:9200/neural/testnn/", json=elasticData)

    if step % step_per_epoch ==0:
        elasticData['phase'] = "test"

        while True:
            precision = 1.0 - math.exp(-step * 0.01) * random.uniform(0,1)
            if precision < 1.0:
                break
        while True:
            recall = 1.0 - math.exp(-step * 0.01) * random.uniform(0,1)
            if recall < 1.0:
                break

        elasticData["precision"] = precision
        elasticData["recall"] = recall
        elasticData["f1"] = 2.0 * (precision * recall) / (precision + recall)

        del elasticData["learning_rate"]
        del elasticData["loss"]

        r = requests.post("http://localhost:9200/neural/testnn/", json=elasticData)
        print(step)
        print(r.status_code, r.reason)
        print(r.text)
