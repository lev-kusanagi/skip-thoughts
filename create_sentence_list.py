import skipthoughts
from nltk import tokenize
import csv
import unicodecsv

# load file
with open('./data/LectureTable.txt', 'r') as f:
  LectureTable_list = list(unicodecsv.reader(f, delimiter='\t'))
# get list of speeches
speeches = []
for entry in LectureTable_list:
    speeches.append(entry[8])

sentences = []
# for each speech, tokenize into list of speech_sentences, extend sentences
for speech in speeches:
    # speech.encode(encoding='UTF-8',errors='replace')

    speech_sentences = tokenize.sent_tokenize(speech)
    sentences.extend(speech_sentences)

with open('results/LectureTable_sentences.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    writer.writerow(sentences)
