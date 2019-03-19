import sys
import os
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
import logging
import coloredlogs
# import re


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
            'Need to download first a NLTK package: tokenize, punkt')
        nltk.download('punkt')

    # try:
    #     stopwords.words('english')
    # except LookupError:
    #     logger.warning('Need to download first a NLTK package: stopwords')
    #     nltk.download('stopwords')
    #
    # try:
    #     WordNetLemmatizer().lemmatize('foo')
    # except LookupError:
    #     logger.warning('Need to download first a NLTK package: wordnet')
    #     nltk.download('wordnet')


def openCorpus(filename):

    with open(filename, 'r') as f:
        str = f.read()
        return str


def createNGrama(tokens, nbrOfNGrams=6):

    Ngrams = ngrams(tokens, nbrOfNGrams)

    return Ngrams

def probabilityPerWord(tokens):
    fdist = FreqDist(tokens)

    # list with the results of the frequency
    list = []
    for k,v in fdist.items():
        list.append({'word': k, 'value': v})

    #  we order the frequencies by value DESC
    list = sorted(list, key=lambda k: k['value'], reverse=True)

    return list


def main():

        downloadNltkDependencies()
        initLogger()

        book = openCorpus("corpus/dracula.txt")

        # with punctuaction marks
        # tokens = word_tokenize(book)

        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(book)

        # we pass the words to lowercase to avoid double entries
        for i in range(0, len(tokens)):
            tokens[i] = tokens[i].lower()

        Ngrams = createNGrama(tokens, 2)

        list = probabilityPerWord(tokens)

        i = 10
        for item in Ngrams:
            logger.info( item )
            i -= 1
            if i < 0:
                break

        sentence = "he is the one"


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning('Keyboard Interrupt... Exiting')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
