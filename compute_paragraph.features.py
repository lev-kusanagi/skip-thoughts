import skipthoughts
from nltk import tokenize
import unicodecsv
import csv
import numpy
from tqdm import tqdm

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

# Define file to compute embeddings for
file = 'T1_L_Sentence'
file_path = './data/' + file + '.csv'

# load paragraph indices from file
paragraph_indices_filename = './data/T2_L.csv'
with open(paragraph_indices_filename, 'r') as f:
    paragraph_indices_list = list(csv.reader(f, delimiter=',')


# def compute_features(vectors):
#     # not implemented yet
#     return    

# for paragraph_indices in paragraph_indices_list: 
                                  
                              
# # Convert array of sentences vectors into array of vectors
# print('Encoding and computing features...')

# batch_size = 4000
# num_batches = len(file_list) / batch_size

# for batch_number in tqdm(range(num_batches)):
#     batch_start_index = batch_number * batch_size
#     if batch_number == num_batches:
#         batch_end_index = len(file_list)
#     else:
#         batch_end_index = batch_start_index + batch_size

#     batch_sentences = file_list[batch_start_index:batch_end_index][1]
#     batch_vectors = encoder.encode(batch_sentences, verbose=False)

#     features = compute_features(batch_vectors)

#     numpy.save("./results/" + str(batch_number) + "_vectors.npy", batch_vectors)
#     # paragraph_indices = list(unicodecsv.reader(open(paragraph_indices_file), 'r'), delimiter=',')
