import json
from datetime import date, datetime
import string
import re
import pandas as pd
import numpy

hashtag_stats = {}
hashtag_cnt = {}
hashtag_names = {}

weight = {}

dataframe = 0

def weight_eval():
	global weight
	global hashtag_stats
	global hashtag_cnt
	w = numpy.array(list(hashtag_cnt.values()))
	w = w.max()

	hashtag_cnt = {k: v/w for (k, v) in hashtag_cnt.items()}
	weight = {hashtag_stats[k]: w for (k,w) in hashtag_cnt.items()}
	
	


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
				if hashtag_stats.get(tag) is None:
					hashtag_stats[tag] = i
					hashtag_names[i] = tag
					i += 1
				if hashtag_cnt.get(tag) is None:
					hashtag_cnt[tag] = 1
				else:
					hashtag_cnt[tag] += 1


def create_dataframe():
	global dataframe
	with open('train_msg.json', 'rb') as fd: 
		messages = json.load(fd) 
	for mes in messages:
		vector = []
		for uniq_tag in hashtag_stats: 
			if check(mes["hashtag"], uniq_tag) == True:
				vector.append(1)
			else:
				vector.append(0)
		mes["hashtag"] = numpy.array(vector, int)
		#mes["hashtag"] = vector
	dataframe = pd.DataFrame(messages)

def metrics(i, j):
	global weight
	global dataframe
	v1 = dataframe.loc[i]['hashtag']
	v2 = dataframe.loc[j]['hashtag']
	diff = 0 
	match = 0
	for k in range(len(v1)):
		if v1[k] != v2[k]: 
			diff += 1
		if (v1[k] == v2[k]) and v1[k] == 1: 
			match += 1
	return diff/(diff + match)

def print_f(i1, i2):
	print('  : ', end='')
	for i in range(i1, i2):
	    print('  %2d '%i, end='')
	print()
	for i in range(i1, i2):
	    print('%2d: '%i, end='')
	    for j in range(i1, i2):
	        print('%.2f '%metrics(i, j), end='')
	    print()


def print_hashtag(idx):
	for i in range(len(dataframe.loc[idx]['hashtag'])):
		if dataframe.loc[idx]['hashtag'][i]:
			print(hashtag_names[i], end=' ')
	print()


def open_file(file_name):
	with open(file_name, 'rb') as fd: 
		messages = json.load(fd)
	return messages

def create_test_and_train():
	test_msg = []
	messages = open_file('channel_messages.json')
	for mes in messages[0:1000]:
		s = {
			"user_id":mes["user_id"], 
			"msg_id":mes["msg_id"], 
			"first_name":mes["first_name"], 
			"last_name":mes["last_name"], 
			"username":mes["username"], 
			"message":mes["message"],
			"hashtag":mes["hashtag"]}
		test_msg.append(s)
	with open('test_msg.json', 'w', encoding='utf8') as outfile:
		json.dump(test_msg, outfile, ensure_ascii=False, indent=4, sort_keys=True)

	train_msg = []
	for mes in messages[1000:11000]:
		s = {
			"user_id":mes["user_id"], 
			"msg_id":mes["msg_id"], 
			"first_name":mes["first_name"], 
			"last_name":mes["last_name"], 
			"username":mes["username"], 
			"message":mes["message"],
			"hashtag":mes["hashtag"]}
		train_msg.append(s)
	with open('train_msg.json', 'w', encoding='utf8') as outfile:
		json.dump(train_msg, outfile, ensure_ascii=False, indent=4, sort_keys=True)



hashtags()

weight_eval()

create_dataframe()

i1 = 0
i2 = 10
for i in range(20):
	print_f(i1, i2)
	i1 += 10
	i2 += 10

#print_f(40, 50)
print_hashtag(33)
print_hashtag(34)
print_hashtag(35)





