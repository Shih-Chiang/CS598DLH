import xml.dom.minidom
import glob
import re
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.tokenize import sent_tokenize


mode = 'test'
if mode == 'train':
    files = glob.glob('data/train*/*.xml')
else:
    files = glob.glob('data/test*/*.xml')


documents = []
for file in files:
    DOMTree = xml.dom.minidom.parse(file)
    TEXT = DOMTree.getElementsByTagName("TEXT")[0]
    TAGS = DOMTree.getElementsByTagName("TAGS")[0]
    T = TEXT.childNodes[0].data

    class Example:
        def __init__(self, start, end, type, text):
            self.start= start
            self.end = end
            self.type = type
            self.text = text

    class Document:
        def __init__(self, text, tags, filename):
            self.text = text
            self.tags = tags
            self.filename = filename

    examples = []
    for tag in TAGS.childNodes:
        if tag.nodeType == 1:
            start = tag.getAttribute('start')
            end = tag.getAttribute('end')
            text = tag.getAttribute('text')
            type = tag.getAttribute('TYPE')
            examples.append(Example(start, end, type, text))
    documents.append(Document(T, examples, file))

def assign_tag(label, text, tag_text, type, first):
    text = text[first:]
    distance = len(tag_text)
    try:
        if distance == 1:
            start = text.index(tag_text[0]) + first
        else:
            start = text.index(tag_text[0]) + first
            second = text.index(tag_text[1]) + first
            if second - start != 1:
                return label, first
    except:
        return label, first

    if distance == 1:
        label[start] = 'B-' + type
    elif distance == 2:
        label[start] = 'B-' + type
        label[start+1] = 'I-' + type
    else:
        label[start] = 'B-' + type
        label[start+distance-1] = 'I-' + type
        for i in range(start+1, start+distance-1):
            label[i] = 'I-' + type
    return label, start + distance


types = set()
res = []
useless_text_num = 0
for document in documents:
    useful_text_flag = True
    text = document.text
    text = re.sub('[\n]+', ' ', text)
    text = re.sub('[ ]+', ' ', text)
    text = list(tokenize(text.strip()))
    word_len = [len(x) for x in text]
    tags = document.tags
    textlen = len(text)
    label = ['O'] * textlen
    first = 0
    for tag in tags:
        tag_text = list(tokenize(tag.text))
        start = int(tag.start)
        end = int(tag.end)
        type = tag.type
        types.add(type)
        label, first = assign_tag(label, text, tag_text, type, first)
    sents = sent_tokenize(' '.join(text))
    length = 0
    for sent in sents:
        words = sent.split()
        end_length = length + len(words)
        for word, lab in zip(words, label[length:end_length]):
            res.append(word + '\t' + lab + '\n')
        res.append(' \n')
        length = end_length

if mode == 'test':
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(res))
else:
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(res))

print('types: ', types)
print('total documents num %d' % (len(documents)))