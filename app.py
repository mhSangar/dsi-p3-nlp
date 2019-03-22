import sys
import os
import logging
import coloredlogs
from nltk import ngrams, download
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
# import re


# Press F to pay respects


logger = logging.getLogger(__name__)


def initLogger(level='INFO'):
    coloredlogs.install(fmt='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S', level=level)
    logger.info('Logger is active')


def downloadNltkDependencies():
    try:
        word_tokenize('foo bar')
    except LookupError:
        logger.warning(
            'Need to download first a NLTK package: punkt')
        download('punkt')

    try:
        stopwords.words('english')
    except LookupError:
        logger.warning('Need to download first a NLTK package: stopwords')
        download('stopwords')


def openCorpus(filename):
    with open(filename, 'r') as f:
        str = f.read()
        return str


def createNGrams(tokens, nbrOfNGrams):
    Ngrams = ngrams(tokens, nbrOfNGrams)

    return Ngrams


def probabilityPerGram(tokens, n, order=False):
    if n == 1:
        ngrams = createNGrams(tokens, n)
        fdist = FreqDist(ngrams)

        return fdist
    
    fdist = None
    fdistMinus1 = None

    ngrams = createNGrams(tokens, n)
    ngramsMinus1 = createNGrams(tokens, n-1)

    fdist = FreqDist(ngrams)
    fdistMinus1 = FreqDist(ngramsMinus1)

    for gram, value in fdist.items():
        # print(gram)
        # print(value)
        # print(fdistMinus1[gram[:-1]])
        fdist[gram] = (value / fdistMinus1[gram[:-1]]) * 100

    # list with the results of the frequency
    gramList = []
    for k, v in fdist.items():
        gramList.append({'gram': k, 'value': v})

    #  we order the frequencies by value DESC
    if order:
        gramList = sorted(gramList, key=lambda k: k['value'], reverse=True)

    return gramList


def removeStopWords(wordList, language='english'):
    stopWords = set(stopwords.words(language))

    filteredWordList = []

    for item in wordList:
        if item.lower() not in stopWords:
            filteredWordList.append(item)

    return filteredWordList


def probabilityOfSentence(sentence, corpusPath, cleanStopWords=False):
    corpus = openCorpus(corpusPath)

    tokenizer = RegexpTokenizer(r'\w+')
    
    sentenceTokens = tokenizer.tokenize(sentence)
    # length of sentence
    n = len(sentenceTokens)

    corpusTokens = tokenizer.tokenize(corpus)

    # clean stop words
    if cleanStopWords:
        corpusTokens = removeStopWords(corpusTokens)

    # we pass the words to lowercase to avoid double entries
    for i in range(0, len(corpusTokens)):
        corpusTokens[i] = corpusTokens[i].lower()

    gramList = probabilityPerGram(corpusTokens, n)
    
    for item in gramList:
        if item['gram'] == tuple(sentenceTokens):
            return item['value']
    
    return 0


def main():
    sentence = sys.argv[1]
    
    p = probabilityOfSentence(sentence, 'corpus/dracula.txt')

    logger.info('sentence: "{}"'.format(sentence))
    logger.info('prob: {:.2f}%'.format(p))


if __name__ == '__main__':
    initLogger()
    downloadNltkDependencies()

    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Keyboard Interrupt... Exiting')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
