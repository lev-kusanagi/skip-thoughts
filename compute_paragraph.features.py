import numpy
from tqdm import tqdm
import csv
filenames_end = '_vectors.npy'
file_path = './'
number_of_files = 251
batch_size = 4000
results = []

sentence_generator = ()

for paragraph_indices in paragraph_indices_list:
    # collect sentences
    sentences = []
    paragraph_indices_range = paragraph_indices[1] - paragraph_indices[0]
    for i in range(paragraph_indices_range):
        sentences.append(sentence_generator.yield) # fix syntax

    # compute features


    
# for batch_number in range(251):

#     vectors = numpy.load(file_path + str(batch_number) + filenames_end)
#     batch_first_sentence_index = batch_number * batch_size
#     batch_last_sentence_index = batch_first_sentence_index + len(vectors)

#     sentence_index = batch_first_sentence_index
#     for paragraph_indices in paragraph_indices_list:
#         if paragraph_indices 

# num_batches = len(sentences) / batch_size

for batch_number in tqdm(range(num_batches)):
    batch_start_index = batch_number * batch_size
    if batch_number == num_batches:
        batch_end_index = len(file_list)
    else:
        batch_end_index = batch_start_index + batch_size

    batch_sentences = file_list[batch_start_index:batch_end_index][1]
    batch_vectors = encoder.encode(batch_sentences, verbose=False)
    # numpy.savetxt("./results/" + file + "_embeddings_batch_"+ str(batch_number) +  ".csv", batch_vectors, delimiter=",")
    numpy.save(str(batch_numer) + "_vectors.npy")
