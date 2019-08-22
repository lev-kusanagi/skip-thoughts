import skipthoughts
from nltk import tokenize
import unicodecsv
import numpy

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

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


# Convert array of sentences vectors into array of vectors
print('encoding...')

batch_size = 2000

for batch_number in range(num_batches):
    batch_start_index = batch_number * batch_size
    if batch_number == num_batches:
        batch_end_index = len(sentences)
    else:
        batch_end_index = batch_start_index + batch_size

    batch_sentences = sentences[batch_start_index:batch_end_index]
    batch_vectors = encoder.encode(batch_sentences)
    numpy.savetxt("./results/LectureTable_embeddings_batch_"+ str(batch_number) +  ".csv", vectors, delimiter=",")
