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
    logger.debug('Logger is active')


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


def openCorpus(filename, test=True):
    if not test:
        with open(filename, 'r') as f:
                str = f.read()
                return str
    else:
        return '''
            And he went back to meet the fox. "Goodbye," he said.

            "Goodbye," said the fox. "And now here is my secret, a very simple secret: It is only with the
            heart that one can see rightly; what is essential is invisible to the eye."

            "What is essential is invisible to the eye," the little prince repeated, so that he would be sure to
            remember. 
        '''


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
        fdist[gram] = (value / fdistMinus1[gram[:-1]]) #* 100

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


def firstWordProb(firstWord, tokens):
    prob = 0
    unigram = createNGrams(tokens, 1)

    fdist = FreqDist(unigram)

    freqWLow = 0
    freqWCap = 0
    freqW = 0

    for gram, value in fdist.items():
        if gram[0] == firstWord.lower():
            freqWLow = value
        if gram[0] == firstWord.capitalize():
            freqWCap = value
        if gram[0] == firstWord:
            freqW = value

    # logger.debug(freqW)
    # logger.debug(freqWLow)
    # logger.debug(freqWCap)

    try:
        prob = freqW / (freqWLow + freqWCap)
    except ZeroDivisionError:
        prob = 0

    return prob


def probabilityOfSentence(sentence, nbrOfNGrams, corpusPath, toLowerCase=False, cleanStopWords=False):
    corpus = openCorpus(corpusPath)

    tokenizer = RegexpTokenizer(r'\w+')
    corpusTokens = tokenizer.tokenize(corpus)
    sentenceTokens = tokenizer.tokenize(sentence)

    if len(sentenceTokens) < nbrOfNGrams:
        logger.warning(
            'Sentence is too short ({nbrTokens} tokens) for the indicated number of ngrams ({nbrOfNGrams}), exiting...'.format(
                nbrTokens=len(sentenceTokens),
                nbrOfNGrams=nbrOfNGrams
        ))
        sys.exit(1)


    # clean stop words
    if cleanStopWords:
        corpusTokens = removeStopWords(corpusTokens)
        sentenceTokens = removeStopWords(sentenceTokens)

    # we pass the words to lowercase to avoid double entries
    if toLowerCase:
        for i in range(0, len(corpusTokens)):
            corpusTokens[i] = corpusTokens[i].lower()
        for i in range(0, len(sentenceTokens)):
            sentenceTokens[i] = sentenceTokens[i].lower()

    gramList = probabilityPerGram(corpusTokens, nbrOfNGrams)
    
    sentenceTuples = []
    for i in range(0, len(sentenceTokens)):
        tup = sentenceTokens[i:i+nbrOfNGrams]
        if len(tup) == nbrOfNGrams:
            sentenceTuples.append(tup)

    firstWord = sentenceTokens[0]
    prob = firstWordProb(firstWord, corpusTokens)
    logger.debug('P{word}: {prob:.5f} %'.format(word=[firstWord], prob=prob))
    for tup in sentenceTuples:
        found = False
        for item in gramList:
            if item['gram'] == tuple(tup):
                logger.debug('P{ngram}: {prob:.5f} %'.format(ngram=tup, prob=item['value'] * 100))
                prob *= item['value']
                found = True
        
        if not found:
            logger.debug('P{ngram}: {prob:.5f} %'.format(ngram=tup, prob=0 * 100))
            prob = 0
    
    return prob


def main():
    nbrOfNGrams = int(sys.argv[1])
    logger.info('NGrams:   {n}'.format(n=nbrOfNGrams))
    sentence = sys.argv[2]
    logger.info('Sentence: "{s}"'.format(s=sentence))

    p = probabilityOfSentence(sentence, nbrOfNGrams, 'corpus/saint-exupery-little-prince.txt')

    logger.info('prob: {:.2f}%'.format(p*100))


if __name__ == '__main__':
    initLogger('DEBUG')
    downloadNltkDependencies()

    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Keyboard Interrupt... Exiting')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
