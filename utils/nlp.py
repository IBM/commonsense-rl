from unidecode import unidecode
import nltk
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class Tokenizer:
    def __init__(self, noun_only_tokens=False, use_stopword=False,
                 ngram=3, extractor=None, openie_url="http://localhost:9000/"):
        """
        :param device:
        :param embedding: can be either 'glove' or 'vocab'
        """
        self.noun_only_tokens = noun_only_tokens
        self.openie_url = openie_url
        self.use_stopword = use_stopword
        self.ngram = ngram
        self.extractor = extractor
        self.nlp_eval = spacy.load("en_core_web_sm")
        self.eng_stopwords = spacy.lang.en.stop_words.STOP_WORDS if use_stopword else None
        self.ignore_list = ['textworld', 'house', 'thing', 'stuff','place', 'yourself', 'kind', 'waste','letdown','us','uk','way','day','many','some', 'location', '-=']
        self.ignore_tags = ['DET', 'PRON', 'PUNCT']

    def _get_word_id(self, word, map2id, unk=True):
        if unk and word not in map2id:
            return -1
        return map2id[word]

    def clean_string(self, text, preprocess=False):
        if preprocess:
            text = self._preprocess(text)
        text = text.lower()
        text = unidecode(text)
        text = re.sub("[^A-Za-z0-9.]+", " ", text)
        return text.strip()

    def _preprocess(self, s, noun_only=False, stopword=False):
        if s is None:
            return ""
        if "$$$$$$$" in s:
            s = s.split("$$$$$$$")[-1]
        if "are carrying:" in s:
            s = " -= inventory =- " + s
        s = s.replace("\n", ' ')
        if s.strip() == "":
            return ""
        s = s.strip()
        if len(s) == 0:
            return ""
        s = s.lower()
        if noun_only:
            pruned_entities = [word.text for word in self.nlp_eval(s) if word.pos_ in ["NOUN", "PROPN", "ADJ"]]
        elif noun_only and stopword:
            pruned_entities = [word.text for word in self.nlp_eval(s) if
                               word.pos_ in ["NOUN", "PROPN", "ADJ"] and not word.is_stop]
        else:
            pruned_entities = [word.text for word in self.nlp_eval(s)]
        # for stop-word: word.is_stop
        pruned_text = " ".join(pruned_entities)
        return pruned_text

    def tokenize(self, text, vocabulary, custom_extractor = None):
        # self.nlp_tokenizer = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        if custom_extractor:
            entities = custom_extractor(text, vocabulary, ngram=self.ngram, stopwords=self.stopwords)
        else:
            entities = self.extractor(text, vocabulary, ngram=self.ngram, stopwords=self.stopwords)
        if self.noun_only_tokens:
            is_noun = lambda pos: pos[:2] == 'NN'
            entities = [word for (word, pos) in nltk.pos_tag(entities) if is_noun(pos)]
        return entities

    def tokenize2(self, text, vocabulary):
        tokens = self.clean_string(text).split()

        if not vocabulary:
            return tokens

        vocabulary = {w for w in vocabulary if "_" in w}

        if len(vocabulary) == 0:
            return tokens

        i = 0
        entities = []
        while i < len(tokens):
            match_found = False
            for j in range(self.ngram, 0, -1):
                if i + j <= len(tokens):
                    entity = "_".join(tokens[i:i + j])
                    if entity in vocabulary:
                        entities.append(entity)
                        i += j
                        match_found = True
                        break
            if not match_found:
                entities.append(tokens[i])
                i += 1
        return entities

    def extract_world_graph_entities(self, text, vocabulary):
        if not text:
            return set()
        processed_text = self._preprocess(text)
        state_doc = self.nlp_eval(processed_text)

        entities = []
        for chunk in state_doc.noun_chunks:
            et = []
            for token in chunk:
                if token.pos_ not in self.ignore_tags:
                    et.append(token.lemma_.lower())
            if et:
                entity = ' '.join(et)
                skip_flag = False
                for ig in self.ignore_list:
                    if ig in entity:
                        skip_flag = True
                if entity in spacy.lang.en.stop_words.STOP_WORDS:
                    skip_flag = True
                if not skip_flag:
                    entities.append(entity)
        text_entities = []
        for entity in entities:
            text_entities.extend(self.extractor(entity, vocabulary))
        return set(text_entities)

    def extract_entity(self,text, vocabulary, noun_only_tokens=None, stopwords = None):
        if not noun_only_tokens:
            noun_only_tokens = self.noun_only_tokens
        if not stopwords:
            stopwords = self.eng_stopwords
        text = self._preprocess(text, noun_only_tokens, stopwords)
        return self.extractor(text, vocabulary, self.ngram, stopwords=stopwords)

    def extract_entity_ids(self, text, vocabulary, unk=True):
        words = self.extract_entity(text, vocabulary)
        # words = self.tokenize(text, vocabulary)
        word_ids = []
        for word in words:
            try:
                index = self._get_word_id(word, vocabulary, unk)
                word_ids.append(index)
            except KeyError:
                    pass
        if not word_ids:
            return [-1]
        return word_ids

