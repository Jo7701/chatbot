import tqdm
from collections import Counter

def grab_data():
	unordered_parent = open('../processed_data/processed_parent.txt', 'r', encoding = 'ISO-8859-1').readlines()
	unordered_reply = open('../processed_data/processed_reply.txt', 'r', encoding = 'ISO-8859-1').readlines()

	parent = [1] * len(unordered_parent)
	reply = [1] * len(unordered_parent)

	for i in tqdm.tqdm(range(len(parent))):
		parent_comment = unordered_parent[i][:-1].split(" ")
		reply_comment = unordered_reply[i][:-1].split(" ")

		parent[int(parent_comment[0])] = parent_comment[1:]
		reply[int(reply_comment[0])] = reply_comment[1:]

	blacklist = sorted({i for i in range(len(parent)) if parent[i] == 1 or len(parent[i]) == 0}|{i for i in range(len(reply)) if reply[i] == 1 or len(reply[i]) == 0}, reverse = True)
	for idx in blacklist:
		del(parent[idx])
		del(reply[idx])

	return parent, reply

print("Loading Data")
parent, reply = grab_data()

print("Creating frequency dictionaries")
parent_freq_dict = Counter(word.lower() for comment in parent for word in comment)
reply_freq_dict = Counter(word.lower() for comment in reply for word in comment)

print("Creating vocabularies")
parent_vocab = {word for word in parent_freq_dict if parent_freq_dict[word] > 5} | {"PAD", "UNK"}
reply_vocab = {word for word in reply_freq_dict if reply_freq_dict[word] > 5} | {"SOS", "EOS", "PAD", "UNK"}

max_enc_time = max(parent, key = len)
max_dec_time = max(reply, key = len)

enc_features = len(parent_vocab)
dec_features = len(reply_vocab)

print("Creating mapping dictionaries")
parent_w2i, reply_w2i = {word:index for index,word in enumerate(parent_vocab)}, {word:index for index, word in enumerate(reply_vocab)}
parent_i2w, reply_i2w = {integer:word for word,integer in parent_w2i.items()}, {integer:word for word,integer in reply_w2i.items()}
