import skipthoughts
from nltk import tokenize
import unicodecsv
import numpy
from tqdm import tqdm

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

# Define file to compute embeddings for
# file = 'LectureTable'
file = 'QATable'
file_path = './data/' + file + '.txt'

# load file
with open(file_path, 'r') as f:
  file_list = list(unicodecsv.reader(f, delimiter='\t'))
# get list of speeches
speeches = []
for entry in file_list:
    speeches.append(entry[8])

sentences = []
# for each speech, tokenize into list of speech_sentences, extend sentences
for speech in speeches:
    # speech.encode(encoding='UTF-8',errors='replace')

    speech_sentences = tokenize.sent_tokenize(speech)
    sentences.extend(speech_sentences)


# Convert array of sentences vectors into array of vectors
print('encoding...')

batch_size = 4000
num_batches = len(sentences) / batch_size

for batch_number in tqdm(range(num_batches)):
    batch_start_index = batch_number * batch_size
    if batch_number == num_batches:
        batch_end_index = len(sentences)
    else:
        batch_end_index = batch_start_index + batch_size

    batch_sentences = sentences[batch_start_index:batch_end_index]
    batch_vectors = encoder.encode(batch_sentences, verbose=False)
    numpy.savetxt("./results/" + file + "_embeddings_batch_"+ str(batch_number) +  ".csv", batch_vectors, delimiter=",")
