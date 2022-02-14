import json
import string
import re
import pandas as pd
import numpy
import nltk
import sys
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas as pd
#import KMeans
from sklearn.cluster import KMeans
from sklearn import metrics

def open_file(file_name):
	with open(file_name, 'rb') as fd: 
		messages = json.load(fd)
	return messages

def normalisation_class(file_name):
	test = open_file(file_name)
	word_test = {}
	for tag, words in test.items():
		for word, cnt in words:
			if word_test.get(word) is None: 
				word_test[word] = {}
			word_test[word][tag] = cnt
	for word in word_test:
		total_cnt = 0	
		for cnt in word_test[word].values():
			total_cnt += cnt
		for tag, cnt in word_test[word].items():
			word_test[word][tag] = cnt / total_cnt

	with open('new_class.json', 'w', encoding='utf8') as outfile:
		json.dump(word_test, outfile, ensure_ascii=False, indent=4, sort_keys=True)
	return word_test, sorted(test.keys())
#
def probabilistic_model(messages, i1, i2):
	vector_terms = {}
	for mes in messages[i1:i2]:
		text = nltk.Text(mes['message'].split())
		fdist = FreqDist(text)
		vector_terms[mes['msg_id']] = fdist.most_common(10)
	
	
	with open('msg.json', 'w', encoding='utf8') as outfile:
		json.dump(vector_terms, outfile, ensure_ascii=False, indent=4, sort_keys=True)


def classification_most_prob_tags(text, all_tags, word_test):
	word_cnt = 0
	tag_prob = {tag:0.0 for tag in all_tags}
	for word in text.split():
		if word_test.get(word) is not None:
			word_cnt += 1
			for tag, prob in word_test[word].items():
				tag_prob[tag] += prob
	for tag, prob in tag_prob.items():
		tag_prob[tag] = prob/word_cnt

	tag_prob = [(v, k) for (k, v) in tag_prob.items()]
	tag_prob.sort(reverse=True)

	#with open('tag_prob.json', 'w', encoding='utf8') as outfile:
	#	json.dump(tag_prob, outfile, ensure_ascii=False, indent=4, sort_keys=True)
	return tag_prob



def classification(text, all_tags, word_test):
	word_cnt = 0
	tag_prob = {tag:0.0 for tag in all_tags}
	for word in text.split():
		if word_test.get(word) is not None:
			word_cnt += 1
			for tag, prob in word_test[word].items():
				tag_prob[tag] += prob
	for tag, prob in tag_prob.items():
		tag_prob[tag] = prob/word_cnt

	#tag_prob_vector = numpy.array([tag_prob[tag] for tag in all_tags], dtype=float)
	tag_prob_vector = {tag:tag_prob[tag] for tag in all_tags}

	#with open('tag_prob.json', 'w', encoding='utf8') as outfile:
	#	json.dump(tag_prob, outfile, ensure_ascii=False, indent=4, sort_keys=True)
	return tag_prob_vector


def classification_old(text, all_tags, word_test):
      word_cnt = 0
      tag_prob = {tag:0.0 for tag in all_tags}
      for word in text.split():
            if word_test.get(word) is not None:
                  word_cnt += 1
                  for tag, prob in word_test[word].items():
                        tag_prob[tag] += prob
      for tag, prob in tag_prob.items():
            tag_prob[tag] = prob/word_cnt

      tag_prob_vector = numpy.array([tag_prob[tag] for tag in all_tags], dtype=float)
      

      with open('tag_prob.json', 'w', encoding='utf8') as outfile:
            json.dump(tag_prob, outfile, ensure_ascii=False, indent=4, sort_keys=True)
      return tag_prob_vector


def print_classificztion(messages, word_test, all_tags, m1, m2, t1, t2):
	i = 0
	for mes in messages[m1:m2]:
		print(i, mes['hashtag'])
		tag_prob = classification_most_prob_tags(mes['message'], all_tags, word_test)
		for prob, tag in tag_prob[t1:t2]:
			print('  ', tag, ':', prob)
		i+=1

def accuracy_classification(messages, word_test, all_tags):
	i = 0
	k = 0
	for mes in messages:
		tag_prob = classification_most_prob_tags(mes['message'], all_tags, word_test)
		for prob, tag in tag_prob[0:1]:
			for mes_tag in mes['hashtag']:
				if mes_tag == tag:
					k+=1
		i+=1
	print('accuracy: %f'%(k/i))

def vector_tag_prob_3_lab(messages, all_tags, word_test, i1, i2):
	V = [classification(mes['message'], all_tags, word_test) for mes in messages[i1:i2]]
	print('    ')
	for i in range(i2):
		print('%5d'%i, end = '')
	print()
	for i in range(i2):
		print('%2d: '%i, end = '')
		for j in range(i2):
			print('%.2f'%(1 - (V[i]*V[j]).sum()), end = ' ')
		print()


def print_v_classes(v_classes):
	for class_idx, v_class in v_classes.items():
		print('CLASS', class_idx, ':')
		#print('  mid:', v_class['mid'])
		#print('  OBJECTS:')
		for v in v_class['members']:
			print('  --------------------------')
			for prob, tag in v['tag_prob'][0:5]:
				print('  ', tag, ':', prob)


def vector(v):
	return v['vector']

KMeans.vector = vector

def KMeans_claster_old(messages, all_tags, word_test, classes, i1, i2):
      V = [{
            'tag_prob': classification_most_prob_tags(mes['message'], all_tags, word_test),
            'vector': classification_old(mes['message'], all_tags, word_test)
            }
            for mes in messages[i1:i2]
            ]
      Vn = numpy.array([v['vector'] for v in V])

      km = KMeans.KMeans(V, classes, vector_len = len(V[0]['vector']), max_a = Vn.max(axis = 0), min_a = Vn.min(axis = 0))
      print_v_classes(km)

def KMeans_claster(messages, all_tags, word_test, classes, i1, i2):
	hashtags()	
	Vv = []
	for mes in messages[i1:i2]:
		d = create_tag_vector(mes, all_tags)
		d['Label'] = mes['msg_id']
		Vv.append(d)
	dataframe_v = pd.DataFrame(Vv)
	
	V = []
	for mes in messages[i1:i2]:
		d = classification(mes['message'], all_tags, word_test)
		d['Label'] = mes['msg_id']
		V.append(d)
	dataframe = pd.DataFrame(V)

	#print(dataframe)
	#print(dataframe_v)
	df = dataframe.join(dataframe_v.set_index('Label'), on='Label', rsuffix='_')
	df['Label'] = df.pop('Label')
	#print(df.describe())

	data_for_cluster = df.iloc[:, 0:-1] 
	labeled = df['Label'] 
	kmeans=KMeans(n_clusters=classes, init='k-means++', max_iter= 300, n_init= 10, random_state= 0) 
	y_kmeans=kmeans.fit_predict(data_for_cluster) 
	#print((data_for_cluster))
	for category, l, in zip(df['Label'], y_kmeans):
		if l == 1:
			print(l, category)
	for category, l, in zip(df['Label'], y_kmeans):
		if l == 1:
			print(l, category)
		

	#print(kmeans.labels_)
	#print(y_kmeans)
	print("Adjusted Rand-Index: %.9f" % metrics.adjusted_rand_score(labeled, kmeans.labels_)) 
	print("Homogeneity: %0.9f" % metrics.homogeneity_score(labeled, kmeans.labels_)) 
	print("Completeness: %0.9f" % metrics.completeness_score(labeled, kmeans.labels_)) 
	print("V-measure: %0.9f" % metrics.v_measure_score(labeled, kmeans.labels_)) 
	print("Silhouette Coefficient: %0.9f" % metrics.silhouette_score(data_for_cluster, kmeans.labels_, sample_size=1000))


	#print_v_classes(km)


hashtag_stats = {}
hashtag_cnt = {}
hashtag_names = {}

def check(tags, uniq_tag): 
	for tag in tags: 
		if tag == uniq_tag: 
			return True 
	return False

def hashtags():
	global hashtag_stats
	global hashtag_cnt
	global hashtag_names
	i = 0
	with open('train_msg.json', 'rb') as fd: 
		messages = json.load(fd) 
		for mes in messages:
			for tag in mes["hashtag"]:
				if len(tag) > 1:
					if hashtag_stats.get(tag) is None:
						hashtag_stats[tag] = i
						hashtag_names[i] = tag
						i += 1
					if hashtag_cnt.get(tag) is None:
						hashtag_cnt[tag] = 1
					else:
						hashtag_cnt[tag] += 1

def create_tag_vector(mes, all_tags):
	vector = {}
	for uniq_tag in all_tags: 
		if uniq_tag in mes["hashtag"]:
			vector[uniq_tag] = 1
		else:
			vector[uniq_tag] = 0
	return vector




def classification_most_prob_tags_for_5(message, all_tags, word_test):
	word_cnt = 0
	tag_prob = {tag:0.0 for tag in all_tags}
	for word in message['message'].split():
		if word_test.get(word) is not None:
			word_cnt += 1
			for tag, prob in word_test[word].items():
				tag_prob[tag] += prob
	v = create_tag_vector(message, all_tags)

	for tag, prob in tag_prob.items():

		tag_prob[tag] = (prob/word_cnt + v[tag]/len(message["hashtag"]))/2

	tag_prob = [(v, k) for (k, v) in tag_prob.items()]
	tag_prob.sort(reverse=True)

	#with open('tag_prob.json', 'w', encoding='utf8') as outfile:
	#	json.dump(tag_prob, outfile, ensure_ascii=False, indent=4, sort_keys=True)
	return tag_prob

def print_classificztion_for_5(messages, word_test, all_tags, m1, m2, t1, t2):
	i = 0
	for mes in messages[m1:m2]:
		print(i, mes['hashtag'])
		tag_prob = classification_most_prob_tags_for_5(mes, all_tags, word_test)
		for prob, tag in tag_prob[t1:t2]:
			print('  ', tag, ':', prob)
		i+=1

messages = open_file('test_msg_without_ends.json')
word_test, all_tags = normalisation_class('class.json')

#print_classificztion(messages, word_test, all_tags, 0, 20, 0, 5)
#accuracy_classification(messages, word_test, all_tags)
#print_classificztion_for_5(messages, word_test, all_tags, 0, 20, 0, 5)


#vector_tag_prob_3_lab(messages, all_tags, word_test, 0, 1)
#KMeans_claster_old(messages, all_tags, word_test, 40, 0, 1000)
KMeans_claster(messages, all_tags, word_test, 40, 0, 1000)














		

