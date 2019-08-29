import skipthoughts
from nltk import tokenize
import unicodecsv
import csv
import numpy
from scipy import spatial
from tqdm import tqdm

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

# Define file to compute embeddings for
file = 'T1_L_Sentence'
file_path = './data/' + file + '.csv'
with open(file_path, 'r') as f:
  file_list = list(csv.reader(f, delimiter=','))
# load paragraph indices from file
paragraph_indices_filename = './data/T2_L.csv'
with open(paragraph_indices_filename, 'r') as f:
    paragraph_indices_list = list(csv.reader(f, delimiter=','))


def get_l1_features(vectors):
    '''Returns mean and std for the l1 distances of vectors'''
    distances = []
    for i in range(len(vectors) - 1):
        distances.append(np.sum(np.abs(vectors[i+1] - vectors[i])))
    return np.mean(distances), np.std(distances)

def get_l2_features(vectors):
    '''Returns mean and std for the l2 distances of vectors'''
    distances = []
    for i in range(len(vectors) - 1):
        distances.append(np.linalg.norm(vectors[i+1] - vectors[i]))
    return np.mean(distances), np.std(distances)

def get_cosine_features(vectors):
    '''Returns mean and std for the cosine similarities of vectors'''
    distances = []
    for i in range(len(vectors) - 1):
        distances.append(spatial.distance.cosine(vectors[i+1], vectors[i]))
    return np.mean(distances), np.std(distances)    

def compute_features(vectors):
    '''Returns a L1_m, L1_std, L2_m, L2_std, cosine_m, cosine_std for the list of input vectors.'''
    l1_features = get_l1_features(vectors)
    l2_features = get_l2_features(vectors)
    cosine_features = get_cosine_features(vectors)
    features = [l1_features, l2_features, cosine_features]
    # flattened_list = [[x for x in y] for y in features]
    return [x for y in features for x in y]

features = []

for paragraph_indices in paragraph_indices_list:
    paragraph_sentences = [x[1] for x in file_list[int(paragraph_indices[0]):int(paragraph_indices[1])]]
    vectors = encoder.encode(batch_sentences, verbose=False)
    
    
                              
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
