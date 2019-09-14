import skipthoughts
import unicodecsv
import numpy as np
from tqdm import tqdm

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
embedding_size = 4800

# Define file to compute embeddings for
file = 'T1_Q_Sentence'
file_path = './data/' + file + '.csv'

# load file
with open(file_path, 'r') as f:
  datastore = list(unicodecsv.reader(f, delimiter=','))

sentences = [x[1] for x in datastore]
vectors = np.empty((0, embedding_size ))
print('Encoding...')

batch_size = 4000
num_batches = len(sentences) / batch_size
output_file_index = 0
checkpoint_frequency = 20

for batch_number in tqdm(range(num_batches)):
  # checkpoint save every 10% progress or so
    batch_start_index = batch_number * batch_size
    if batch_number == num_batches - 1:
        batch_end_index = len(sentences)
    else:
        batch_end_index = batch_start_index + batch_size
        
    batch_sentences = sentences[batch_start_index:batch_end_index]
    batch_vectors = encoder.encode(batch_sentences, verbose=False)
    vectors = np.vstack([vectors, batch_vectors])
    if (batch_number + 1) % checkpoint_frequency == 0:
      np.save('qatable_sentences_embeddings_' + str(batch_size * checkpoint_frequency * output_file_index) + '_to_' + str(batch_size * checkpoint_frequency * (output_file_index + 1)), vectors)
      vectors = np.empty((0, embedding_size ))
      output_file_index += 1
    if batch_number == num_batches - 1:
      np.save('qatable_sentences_embeddings_' + str(batch_size * checkpoint_frequency * output_file_index) + '_to_' + str(batch_end_index), vectors)

 
