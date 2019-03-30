import tqdm

def grab_data():
	unordered_parent = open('../data/processed_parent.txt', 'r').readlines()
	unordered_reply = open('../data/processed_reply.txt', 'r').readlines()

	parent = [1] * len(unordered_parent)
	reply = [1] * len(unordered_parent)

	for i in tqdm.tqdm(range(len(parent))):
		parent_comment = unordered_parent[i][:-1].split(" ")
		reply_comment = unordered_reply[i][:-1].split(" ")
		
		parent[int(parent_comment[0])] = parent_comment[1:]
		reply[int(reply_comment[0])] = reply_comment[1:]
	return parent, reply

parent, reply = grab_data()

print(parent[0])

print("Creating Parent Reply Vocab")

parent_vocab = set()
for i, comment in enumerate(parent):
	try:	
		for word in comment:
			parent_vocab.add(word)
	except:
		print(comment)
		print("INDEX:", i)
		exit()
print(len(parent_vocab))
