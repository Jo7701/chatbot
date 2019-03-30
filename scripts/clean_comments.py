import langid
from tqdm import tqdm

def num_consec(comment):
    max_rep = 1
    counter = 1
    prev_char = comment[0]
    for i,c in enumerate(comment):
        if c == prev_char and i != 0:
            counter += 1
        else:
            if counter > max_rep:
                max_rep = counter
            counter = 0
            prev_char = c
    return max(counter, max_rep)

def valid_comment(comment):
    if len(comment.split()) and len(comment)/len(comment.split()) > 10:
        return False
    if comment.count('http') > 5:
        return False
    if num_consec(comment) > 5:
        return False
    if langid.classify(comment)[0] != 'en':
        return False
    return True

valid_indeces = []

print("Loading Data")
parent = open('../data/parent.txt', 'r').readlines()
reply = open('../data/reply.txt', 'r').readlines()

for i in tqdm(range(len(parent))):
    if valid_comment(parent[i]) and valid_comment(reply[i]):
        valid_indeces.append(i)

print("Num valid pairs:", len(valid_indeces))
#
# file = open('valid_indeces.txt', 'w')
# for index in valid_indeces:
#     file.write(str(index) + '\n')
