import skipthoughts
from nltk import tokenize
import unicodecsv
import csv
import numpy
from tqdm import tqdm

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

# Define file to compute embeddings for
# file = 'LectureTable'
file = 'T1_L_Sentence'
file_path = './data/' + file + '.csv'

# load file
with open(file_path, 'r') as f:
  file_list = list(csv.reader(f, delimiter=','))



# Convert array of sentences vectors into array of vectors
print('encoding...')

batch_size = 4000
num_batches = len(file_list) / batch_size

for batch_number in tqdm(range(num_batches)):
    batch_start_index = batch_number * batch_size
    if batch_number == num_batches:
        batch_end_index = len(file_list)
    else:
        batch_end_index = batch_start_index + batch_size

    batch_sentences = file_list[batch_start_index:batch_end_index][1]
    batch_vectors = encoder.encode(batch_sentences, verbose=False)
    # numpy.savetxt("./results/" + file + "_embeddings_batch_"+ str(batch_number) +  ".csv", batch_vectors, delimiter=",")
    numpy.save("./results/" + str(batch_number) + "_vectors.npy", batch_vectors)
