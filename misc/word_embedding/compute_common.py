import random
from scipy import spatial
import numpy as np

def build_common(vocab_file1, vocab_file2, num_words=0):
    word_set = set()
    with open(vocab_file1, "r") as inFile:
        for line in inFile:
            word = line.split(" ")[0]
            if word.isalpha():
                word_set.add(word)

    common_set = set()
    with open(vocab_file2, "r") as inFile:
        for line in inFile:
            word = line.split(" ")[0]
            if word.isalpha() and word in word_set:
                common_set.add(word)
                if num_words > 0 and len(common_set) == num_words:
                    return common_set 
    return common_set


def compute_similarity_mat(vector_mat, reference_words):
    relation_mat = {}
    for k, v in vector_mat.iteritems():
        similarity_vec = []
        for word in reference_words:
            ref_vec = vector_mat[word]
            similarity = 1 - spatial.distance.cosine(ref_vec,v)
            similarity_vec.append(similarity)
        relation_mat[k] = similarity_vec
    return relation_mat

def load_compute(vector_file, word_set, reference_words):
    vector_mat = {} 
    with open(vector_file, "r") as inFile:
        for line in inFile:
            strs = line.rstrip("\n").split(" ") 
            word = strs[0]
            if word in word_set: 
                vector = map(float, strs[1:])
                vector_mat[word] = vector
    return compute_similarity_mat(vector_mat, reference_words)

def compute_z_score(common_set, matrix1, matrix2):
    word_lst = []  
    diff_lst = []
    for word in common_set:
        vec1 = np.array(matrix1[word])
        vec2 = np.array(matrix2[word])
        diff = np.sum(np.absolute(vec1 - vec2))
        word_lst.append(word)
        diff_lst.append(diff)
    diff_lst = np.array(diff_lst)
    diff_lst = (diff_lst - diff_lst.mean())/diff_lst.std() 
    return np.stack((np.array(word_lst), diff_lst))


def main():
    vocab_file1 = "baseVocab.txt"
    vocab_file2 = "swappedVocab.txt"
    vector_file1 = "baseVector.txt"
    vector_file2 = "swappedVector.txt"

    reference_size = 1000       

    common_set = build_common(vocab_file1, vocab_file2)
    reference_words = random.shuffle(list(common_set))[:reference_size]

    vector_mat1 = load_compute(vector_file1, common_set, reference_words)
    vector_mat2 = load_compute(vector_file2, common_set, reference_words)
    z_scores = compute_z_score(common_set, vector_mat1, vector_mat2)

if __name__ == "__main__" : main()
