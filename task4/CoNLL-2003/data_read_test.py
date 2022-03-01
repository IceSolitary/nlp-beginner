with open("train.txt", 'r', encoding='utf-8') as f:
    contents = []
    tokens = []
    labels = []
    for line in f:
        line = line.strip('\n')

        if line == '-DOCSTART- -X- -X- O':
            pass

        elif line == "":
            if len(tokens):
                contents.append([tokens, labels])
                tokens = []
                labels = []

        else:
            content = line.split(" ")
            tokens.append(content[0])
            labels.append(content[3])


max_len = 0
count = 0
print(len(contents))
for content in contents:
    if len(content[0]) > max_len:
        max_len = len(content[0])
    if len(content[0]) >20:
        count +=1
print(count)
print(max_len)


