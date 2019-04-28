import bz2, json

def decompress(outfile):
    file = open(outfile, 'w')
    for index, line in enumerate(bz2.open("redditdata.bz2")):
        if not index%1000000:
            print("Processed {:,}/53,000,000 comments".format(index))
        line = line.decode('utf-8')
        file.write(line)
    file.close()

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
    if len(comment) < 1: return False
    if len(comment) > 1000: return False
    if comment == '[removed]' or comment == '[deleted]': return False
    try:
        comment.encode('utf-8').decode('ascii')
    except:
        return False
    if num_consec(comment) > 5: return False
    if len(comment.split()) > 50: return False
    if comment.count('http') > 5: return False
    return True

def write_data():
    data = {} #{parent_id:[parent_body, reply_body, reply_score]}
    pairs = {}
    num_pairs = 0

    for index, comment in enumerate(open('decompressed_data.txt', 'r')):
        if not index%1000000:
            print("Processed {:,}/53,000,000 comments".format(index))
            print("Number of comment pairs: {:,}\n".format(num_pairs))

        comment = json.loads(comment)
        comment_id = comment['name']
        parent_id = comment['parent_id']
        score = comment['score']
        body = comment['body'].replace("\n", " ").replace("\r", " ").replace('"', "'")

        if not valid_comment(body):
            continue

        if parent_id in data:
            if isinstance(data[parent_id], list):
                if score > data[parent_id][2]:
                    data[parent_id][1] = body
                    data[parent_id][2] = score
            else:
                data[parent_id] = [data[parent_id], body, score]
                num_pairs += 1
        else:
            if score > 3:
                data[comment_id] = body

    print("Writing Parent Reply Comments to File")

    parent_file = open('parent.txt', 'w')
    reply_file = open('reply.txt', 'w')

    for parent_id in data:
        if isinstance(data[parent_id], list):
            parent_file.write(data[parent_id][0]+"\n")
            reply_file.write(data[parent_id][1]+"\n")

write_data()
