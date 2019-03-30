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

parent, reply = grab_data()

parent_freq_dict = Counter(word.lower() for comment in parent for word in comment)
reply_freq_dict = Counter(word.lower() for comment in reply for word in comment)

parent_vocab = {word for word in parent_freq_dict if parent_freq_dict[word] > 5} | {"PAD", "UNK"}
reply_vocab = {word for word in reply_freq_dict if reply_freq_dict[word] > 5} | {"SOS", "EOS", "PAD", "UNK"}

print("Orig Parent Vocab Size:", len(parent_freq_dict))
print("New Parent Vocab Size:", len(parent_vocab))
print("Orig Reply Vocab Size:", len(reply_freq_dict))
print("New Reply Vocab Size:", len(reply_vocab))
