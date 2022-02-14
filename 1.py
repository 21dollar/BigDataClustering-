import configparser
import json
from telethon.sync import TelegramClient
from telethon import connection
from datetime import date, datetime
from telethon.tl.functions.channels import GetParticipantsRequest
from telethon.tl.types import ChannelParticipantsSearch
from telethon.tl.functions.messages import GetHistoryRequest
import string
import re
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

config = configparser.ConfigParser()
config.read("config.ini")

api_id   = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']
username = config['Telegram']['username']

client = TelegramClient(username, api_id, api_hash)

hashtag_stats = {}

def hashtag_devide(text):
	global hashtag_stats
	text = re.sub('[Ёё]', 'е', text)
	tags = [tag[1:].lower() for tag in re.findall(r'#\w+', text)]
	if 'помогу' in tags:
		tags.remove('помогу')
	else:
		return [], ''
	if 'ищу' in tags:
		tags.remove('ищу')

	no_tags = ' '.join(re.sub(r'#\w+|[^\w ]|\d', ' ', text).lower().split())
	return tags, no_tags

class DateTimeEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, datetime):
			return o.isoformat()
		if isinstance(o, bytes):
			return list(o)
		return json.JSONEncoder.default(self, o)	

async def main():
	channel = await client.get_entity("https://t.me/chat_frilans")
	
	offset_msg = 0    
	limit_msg = 100   

	all_messages = []   
	total_messages = 0
	total_count_limit = 11000  

	while total_messages < total_count_limit:
		history = await client(GetHistoryRequest(
			peer=channel,
			offset_id=offset_msg,
			offset_date=None, add_offset=0,
			limit=limit_msg, max_id=0, min_id=0,
			hash=0))
		if not history.messages:
			break
		user_dict = { }
		for user in history.users:
		    user_dict[user.id] = {\
		    	"first_name": user.first_name.lower() if user.first_name else '', \
			    "last_name": user.last_name.lower() if user.last_name else '', \
			    "username": user.username.lower() if user.username else ''}
		messages = history.messages
		for message in messages:
			mes = message.to_dict()
			try:
				tags, no_tags = hashtag_devide(mes["message"])
				if not tags or user_dict[mes["from_id"]["user_id"]]["username"][-3:] == 'bot':
					continue

				mes = {	"date":mes["date"], 
							"user_id":mes["from_id"]["user_id"], 
							"msg_id":mes["id"], 
							"first_name":user_dict[mes["from_id"]["user_id"]]["first_name"], 
							"last_name":user_dict[mes["from_id"]["user_id"]]["last_name"], 
							"username":user_dict[mes["from_id"]["user_id"]]["username"], 
							"message":no_tags,
							"hashtag":tags}
				all_messages.append(mes)
			except Exception as exc:
				print(exc)
				pass
		offset_msg = messages[len(messages) - 1].id
		total_messages = len(all_messages)
		print('%5d/%5d'%(total_messages, total_count_limit), end='\r')
		if total_count_limit != 0 and total_messages >= total_count_limit:
			break

	with open('channel_messages.json', 'w', encoding='utf8') as outfile:
		json.dump(all_messages, outfile, ensure_ascii=False, indent=4, sort_keys=True, cls=DateTimeEncoder)



with client:
    client.loop.run_until_complete(main())



#l = [(v,k) for k,v in hashtag_stats.items()]
#l.sort()
#for i in l:
#    print(i)
