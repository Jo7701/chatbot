def clean_url(comment):
    http_index = comment.find("http")
    while http_index != -1:
        if http_index != 0 and comment[http_index - 1] == '(':
            start_url = http_index - 1
            end_url = comment.find(")", http_index)
            if end_url == -1:
                end_url = comment.find(' ', start_url)
            comment = comment[:start_url] + comment[end_url+1:]

            if start_url-1 >= 0 and comment[start_url-1] == ']':
                for x in range(start_url - 1, -1, -1):
                    if comment[x] == '[': break
                comment = comment[:start_url-1] + comment[start_url:]
                comment = comment[:x] + comment[x+1:]
            elif start_url - 2 >= 0 and comment[start_url-2] == ']':
                for x in range(start_url-2, -1, -1):
                    if comment[x] == '[': break
                comment = comment[:start_url-2] + comment[start_url:]
                comment = comment[:x] + comment[x+1:]
        else:
            start_url = http_index
            end_url = comment.find(" ", start_url)
            comment = comment[:start_url] + comment[end_url:]

        http_index = comment.find("http")

    return comment

def clean_comment(comment):
    try:
        comment = comment.replace("-&gt;", "")
    except:
        pass
    comment = clean_url(comment)
    return comment

def tokenize(comment, nlp, comment_type):
    comment = clean_comment(comment)
    doc = nlp(comment)
    toRet = []
    for token in doc:
        if comment_type == 'p':
            if not token.is_space and token.lemma_ != '-PRON-':
                toRet.append(token.lemma_)
            elif token.lemma_ == '-PRON-':
                toRet.append(token.text)
        else:
            toRet.append(token.text)
    return toRet
