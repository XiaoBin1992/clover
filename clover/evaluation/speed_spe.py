import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process some JSON files.')
parser.add_argument('--path', type=str, help='Path to the speculative decoding JSONL file')
parser.add_argument('--base_path', type=str, help='The path to the base .jsonl file', default='')

args = parser.parse_args()

data = []
with open(args.path, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)

speeds = [[], [], [], [], [], []]
for i, datapoint in enumerate(data):
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    if i < 80:
        speeds[0].append(tokens/times)
    elif i < 160:
        speeds[1].append(tokens/times)
    elif i < 240:
        speeds[2].append(tokens/times)
    elif i < 320:
        speeds[3].append(tokens/times)
    elif i < 400:
        speeds[4].append(tokens/times)
    elif i < 480:
        speeds[5].append(tokens/times)
    else:
        break


if args.base_path != "":
    data = []
    with open(args.base_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds0 = [[], [], [], [], [], []]
    for i, datapoint in enumerate(data):
        qid=datapoint["question_id"]
        answer=datapoint["choices"][0]['turns']
        tokens=sum(datapoint["choices"][0]['idxs'])
        times = sum(datapoint["choices"][0]['wall_time'])
        if i < 80:
            speeds0[0].append(tokens/times)
        elif i < 160:
            speeds0[1].append(tokens/times)
        elif i < 240:
            speeds0[2].append(tokens/times)
        elif i < 320:
            speeds0[3].append(tokens/times)
        elif i < 400:
            speeds0[4].append(tokens/times)
        elif i < 480:
            speeds0[5].append(tokens/times)
        else:
            break

for i in range(6):
    print('idx: ', i)
    print('speed',np.array(speeds[i]).mean())
    if args.base_path != "":
        print('speed0',np.array(speeds0[i]).mean())
        print("ratio",np.array(speeds[i]).mean()/np.array(speeds0[i]).mean())
