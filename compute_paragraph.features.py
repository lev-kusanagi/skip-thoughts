import skipthoughts
from nltk import tokenize
import unicodecsv
import csv
import numpy
from scipy import spatial
from tqdm import tqdm
from sklearn.manifold import TSNE

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
    cosine_features = get_cosine_features(vectors)
    l1_features = get_l1_features(vectors)
    l2_features = get_l2_features(vectors)
    features = [l1_features, l2_features, cosine_features]
    # flattened_list = [[x for x in y] for y in features]
    return [x for y in features for x in y]

features = []

for paragraph_indices in paragraph_indices_list:
    paragraph_sentences = [x[1] for x in file_list[int(paragraph_indices[0]):int(paragraph_indices[1])]]
    paragraph_features = []
    vectors = encoder.encode(batch_sentences, verbose=False)
    
    # 1. first 24k
    vectors_1st_24 = [x[:2400] for x in vectors]
    paragraph_features.extend(compute_features[vectors_1st_24])
    vectors_last_24 = [x[2400:] for x in vectors]
    paragraph_features.extend(compute_features[vectors_last_24])
    paragraph_features.extend(compute_features[vectors])
    tsne_100 = TSNE(n_components = 100).fit_transform(vectors)
    paragraph_features.extend(compute_features[tsne_100])
    tsne_200 = TSNE(n_components = 200).fit_transform(vectors)
    paragraph_features.extend(compute_features[tsne_200])
    tsne_600 = TSNE(n_components = 600).fit_transform(vectors)
    paragraph_features.extend(compute_features[tsne_600])
    
    features.append(paragraph_features)
