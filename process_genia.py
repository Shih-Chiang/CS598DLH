import json


def assign_tag(label, start, end, type):
    distance = end - start
    if distance == 1:
        label[start] = 'B-' + type
    else:
        label[start] = 'B-' + type
        for i in range(start+1, end):
            label[i] = 'I-' + type
    return label

res = []
with open('genia/genia_test_context.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for line in data:
        w = line['tokens']
        entities = line['entities']
        label = ['O' for _ in range(len(w))]
        for entity in entities:
            label = assign_tag(label, entity['start'], entity['end'], entity['type'])

        for word, l in zip(w, label):
            res.append(word.lower() + '\t' + l + '\n')
        res.append(' \n')

with open('genia/genia_test.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(res))