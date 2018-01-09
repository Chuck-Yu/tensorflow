# label.pbtxt -> labels.txt

import re

labels = open("labels.txt", "wb")
label_pb = open("label.pbtxt", "r")

buff = label_pb.read()  # re cannot input direct from open operation.

str_ids = re.findall('id: (.*?)\n', buff, re.S) # find all ids.
str_labels = re.findall('display_name: "(.*?)"', buff, re.S)    # find all lables.

int_ids = map(eval, str_ids)    # convert all strings to a inter list.

label_list = ['NA' for x in range(0, max(int_ids))] # initialize a list with 'NA'
i = 0
for each_id in str_ids:
    label_list[int(each_id) - 1] = str_labels[i];
    i += 1

for j in range(0, max(int_ids)):
    labels.write(label_list[j] + '\n')

labels.close()
