import multiprocessing
import os
import spacy
import process
import time

def split():
    if os.path.exists('../split_data'):
        return True

    os.makedirs('../split_data')
    os.makedirs('../split_data/parent')
    os.makedirs('../split_data/reply')

    parent = open('../data/parent.txt', 'r').readlines()
    reply = open('../data/reply.txt', 'r').readlines()
    valid_indeces = open('../data/valid_indeces.txt', 'r').readlines()

    parent = [parent[int(i)] for i in valid_indeces]
    reply = [reply[int(i)] for i in valid_indeces]

    chunk_size = len(parent)//multiprocessing.cpu_count()

    for num in range(multiprocessing.cpu_count()):
        parent_file = open('../split_data/parent/chunk'+str(num)+'.txt', 'w')
        reply_file = open('../split_data/reply/chunk'+str(num)+'.txt', 'w')
        for idx in range(chunk_size):
            parent_file.write(str(chunk_size*num+idx) + " " + parent[chunk_size*num+idx])
            reply_file.write(str(chunk_size*num+idx) + " " + reply[chunk_size*num+idx])

def process_file(filename):
    nlp = spacy.load('en')
    toRet = []
    index = 0
    for comment in open(filename, 'r'):
        if index % 10000 == 0:
            print("Processed", index, "comments from", filename)
        toRet.append(process.tokenize(comment, nlp))
        index += 1
    return toRet

def write_data(results, filename):
    with open(filename, 'w') as f:
        for chunk in results:
            for comment in chunk:
                f.write(" ".join(comment)+"\n")
split()

parent_files = ["../split_data/parent/"+str(file) for file in os.listdir('../split_data/parent')]
reply_files = ["../split_data/reply/"+str(file) for file in os.listdir('../split_data/reply')]

pool = multiprocessing.Pool(multiprocessing.cpu_count())
print("PROCESSING PARENT")
parent_results = pool.map(process_file, parent_files)

print("\n\nPROCESSING REPLY")
reply_results = pool.map(process_file, reply_files)
pool.close()

write_data(parent_results, "../data/processed_parent.txt")
write_data(reply_results, "../data/processed_reply.txt")
