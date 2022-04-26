"""i2b2数据预处理程序"""
 import xml.dom.minidom
import glob
import re
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.tokenize import sent_tokenize

# 训练集或测试集文件处理
mode = 'test'
if mode == 'train':
    files = glob.glob('data/train*/*.xml')
else:
    files = glob.glob('data/test*/*.xml')

# 存储所有文章
documents = []
for file in files:
    # 解析xml文件
    DOMTree = xml.dom.minidom.parse(file)
    # 获取TEXT节点
    TEXT = DOMTree.getElementsByTagName("TEXT")[0]
    # 获取TAGS节点
    TAGS = DOMTree.getElementsByTagName("TAGS")[0]
    # 调用data获取TEXT字符串
    T = TEXT.childNodes[0].data

    class Example:
        # 用于存储一篇文章中一个标签的4个属性
        def __init__(self, start, end, type, text):
            self.start= start
            self.end = end
            self.type = type
            self.text = text

    class Document:
        # 用于存储一篇文章中的所有标签属性以及文章文本
        def __init__(self, text, tags, filename):
            self.text = text
            self.tags = tags
            self.filename = filename

    examples = []
    for tag in TAGS.childNodes:
        # type 等于1代表有节点属性，否则没有
        if tag.nodeType == 1:
            start = tag.getAttribute('start')
            end = tag.getAttribute('end')
            text = tag.getAttribute('text')
            type = tag.getAttribute('TYPE')
            examples.append(Example(start, end, type, text))
    documents.append(Document(T, examples, file))

def assign_tag(label, text, tag_text, type, first):
    # 标签个数
    text = text[first:]
    distance = len(tag_text)
    try:
        # 存在错误标注的情况，这种情况就跳过
        if distance == 1:
            start = text.index(tag_text[0]) + first
        else:
            start = text.index(tag_text[0]) + first
            second = text.index(tag_text[1]) + first
            if second - start != 1:
                return label, first
    except:
        return label, first
    # 使用BIO进行标注
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
    # 文本预处理，删除\n,合并空格，并使用nltk的分词包进行标点和文字的分词
    text = re.sub('[\n]+', ' ', text)
    text = re.sub('[ ]+', ' ', text)
    text = list(tokenize(text.strip()))
    word_len = [len(x) for x in text]
    tags = document.tags
    textlen = len(text)
    label = ['O'] * textlen
    first = 0
    for tag in tags:
        # 标签也同样进行分词
        tag_text = list(tokenize(tag.text))
        start = int(tag.start)
        end = int(tag.end)
        type = tag.type
        types.add(type)
        # 对预设置好的全为O的标签进行再分配
        label, first = assign_tag(label, text, tag_text, type, first)
    # 一篇文章的标签分配好了之后，按照字与标签的格式进行存储,同时进行分句
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