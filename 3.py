import json
from datetime import date, datetime
import string
import re
import pandas as pd
import numpy
import nltk
import sys
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer

end_tree = {'а': [1, {'л': [1, {'я': [3, {}]}]}], 'е': [1, {'е': [2, {}], 'и': [2, {}], 'о': [2, {}], 'ы': [2, {}], 'т': [1, {'и': [3, {}]}]}], 'и': [1, {'м': [2, {'и': [3, {}]}], 'л': [1, {'я': [3, {}]}]}], 'м': [1, {'а': [2, {}], 'е': [2, {}], 'и': [2, {}], 'о': [2, {}], 'у': [2, {}]}], 'о': [1, {'г': [1, {'о': [3, {}]}]}], 'у': [1, {'м': [1, {'о': [3, {}]}]}], 'ы': [1, {}], 'ь': [1, {'ш': [2, {'и': [3, {}]}], 'т': [1, {'я': [3, {}]}]}], 'ю': [1, {'е': [2, {}], 'и': [2, {}], 'о': [2, {}], 'у': [2, {}]}], 'я': [1, {'а': [2, {}], 'м': [2, {'у': [3, {}]}], 'я': [2, {}]}], 'в': [0, {'о': [2, {}]}], 'й': [0, {'е': [2, {}], 'и': [2, {}], 'й': [2, {}], 'о': [2, {}], 'ы': [2, {}]}], 'л': [0, {'а': [2, {}], 'е': [2, {}], 'о': [2, {}], 'у': [2, {}], 'я': [2, {}]}], 'с': [0, {'а': [2, {}]}], 'т': [0, {'а': [2, {}], 'е': [2, {}], 'и': [2, {}], 'с': [2, {}], 'у': [2, {}], 'ю': [2, {}], 'я': [2, {}]}], 'х': [0, {'а': [2, {}], 'е': [2, {}], 'и': [2, {}], 'у': [2, {}]}]}

all_words = {}
all_words_cnt = {}

def open_file(file_name):
	with open(file_name, 'rb') as fd: 
		messages = json.load(fd)
	return messages

def nonend(word):
    word_len = len(word)
    if word_len < 3:
        return word
    end_offs = 0
    d = end_tree
    for i in range(word_len - 1, word_len - 4, -1):
        try:
            end_offs, d = d[word[i]]
        except:
            return word[:word_len - end_offs]
    return word[:word_len - end_offs]


def clear(messages):
	russian_stopwords = stopwords.words("russian")
	for mes in messages:
		mes["message"] = mes["message"].replace('_', '')
		mes["message"] = mes["message"].replace('ℹ', '')
		mes["message"] = mes["message"].replace('①', '')
		mes["message"] = mes["message"].replace('②', '')
		mes["message"] = mes["message"].replace('③', '')
		mes["message"] = mes["message"].replace('ᝰ', '')
		mes["message"] = ' '.join(nonend(word) for word in mes["message"].split() if word not in russian_stopwords)

	with open('train_msg_without_ends.json', 'w', encoding='utf8') as outfile:
		json.dump(messages, outfile, ensure_ascii=False, indent=4, sort_keys=True)
	return messages


def create_word_vector(messages):
	for mes in messages:
		text = nltk.Text(mes['message'])
		fdist = FreqDist(text)
		mes['word_vector'] = [fdist.get(uniq_word, 0) for uniq_word in all_words]
	return messages

def get_words(new_messages):
	global all_words
	global all_words_cnt
	i = 0
	for mes in new_messages:
		for word in mes["message"]:
			if all_words.get(word) is None:
				all_words[word] = i
				i += 1
			if all_words_cnt.get(word) is None:
				all_words_cnt[word] = 1
			else:
				all_words_cnt[word] += 1
	l = [(v,k) for k,v in all_words.items()]
	l.sort()
	
	with open('words.json', 'w', encoding='utf8') as outfile:
		json.dump(l, outfile, ensure_ascii=False, indent=4, sort_keys=True)

def check(tags, uniq_tag): 
	for tag in tags: 
		if tag == uniq_tag: 
			return True 
	return False

def vector_model(messages, i1, i2):
	vectorizer = TfidfVectorizer(analyzer='word', min_df=0.0) 
	X = vectorizer.fit_transform(mes['message'] for mes in messages[i1:i2])
	vector_terms = vectorizer.get_feature_names()
	print(X)
	print(vector_terms)

	list_tfidf = []
	m = X.toarray() 
	print(m)
	return vector_terms
		
def print_m(matrix):
	print('  : ', end='')
	for i in range(len(matrix)):
	    print('%2d '%i, end='')
	print()
	for i in range(len(matrix)):
	    print('%2d: '%i, end='')
	    for j in range(len(matrix)):
	        print('%2d '%matrix[i, j], end='')
	    print()

def print_mf(matrix):
	print('  : ', end='')
	for i in range(len(matrix)):
	    print('  %2d '%i, end='')
	print()
	for i in range(len(matrix)):
	    print('%2d: '%i, end='')
	    for j in range(len(matrix)):
	        print('%.1f '%matrix[i, j], end='')
	    print()

def matrix_model(messages, i1, i2):
	vector_terms = vector_model(messages, i1, i2)
	term_idx = {term:i for term,i in zip(vector_terms, range(len(vector_terms)))}
	for mes in messages[i1:i2]:
		matrix = numpy.zeros((len(vector_terms), len(vector_terms)), dtype=int)
		words = [word for word in mes['message'].split() if len(word) > 1] 
		for i in range(len(words)-1):
			matrix[term_idx[words[i]], term_idx[words[i+1]]] += 1
	numpy.set_printoptions(threshold=sys.maxsize)
	print_m(matrix)
	print(term_idx)




def probabilistic_model(messages, i1, i2):
	vectorizer = TfidfVectorizer(analyzer='word', min_df=0.0) 
	X = vectorizer.fit_transform(mes['message'] for mes in messages[i1:i2])
	vector_terms = vectorizer.get_feature_names()

	term_idx = {term:i for term,i in zip(vector_terms, range(len(vector_terms)))}
	
	tag_vectorizer = TfidfVectorizer(analyzer='word', min_df=0.0) 
	tag_X = tag_vectorizer.fit_transform(tag for mes in messages[i1:i2] for tag in mes['hashtag'])
	term_tag = {tag:[] for tag in tag_vectorizer.get_feature_names() if len(tag) > 1}
	#print(term_tag.keys())

	
	for mes in messages[i1:i2]:
		words = mes['message'].split()
		tags = [tag for tag in mes['hashtag'] if len(tag) > 1] 
		for tag in tags:
			term_tag[tag] += words

	for tag, words in term_tag.items():
		text = nltk.Text(words)
		fdist = FreqDist(text)
		term_tag[tag] = fdist.most_common(10)
			
	#for k, v in term_tag.items():
	#	print(k, v)		

	with open('class.json', 'w', encoding='utf8') as outfile:
		json.dump(term_tag, outfile, ensure_ascii=False, indent=4, sort_keys=True)






messages = open_file('train_msg.json')
new_messages = clear(messages)
#vector_model(new_messages, 0, 2)
#matrix_model(new_messages, 8, 10)
#probabilistic_model(new_messages, 0, 10000)


