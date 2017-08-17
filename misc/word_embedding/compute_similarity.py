#  This file is used to compute the absolute difference between two embedding 
#  similariy matrices. Three files are needed as input arguments:
#  1. a word file containing a list of words used as reference to compute the simialrity matrix.
#  2. two vector files in which each word vector is computed using the GloVe embedding.
#
#  License: BSD
#  Author: Sen Jia
#


from scipy import spatial
import numpy as np

def build_similarity_matrix():

    def load_vector(filename, ref_lst):
        vectors = {}
        with open(filename,"r") as inFile:
            for line in inFile:
                 strs = line.rstrip("\n").split(" ")
                 if strs[0] in ref_lst:
                     vector = map(float,strs[1:])
                     vectors[strs[0]] = vector
        return vectors

              
    def compute_similarity_matrix(vectors, reference_lst):
        similarity_mat = []
        vec_len = len(reference_lst) 
        similarity_mat = np.ones((vec_len, vec_len))
        for i in range(vec_len):
            vec1 = vectors[reference_lst[i]]
            for j in range(i+1, vec_len):
                vec2 = vectors[reference_lst[j]]
                similarity = 1 - spatial.distance.cosine(vec1, vec2) 
                similarity_mat[i][j] = similarity
      
        for i in range(vec_len):
            for j in range(0, i):
               similarity_mat[i][j] = similarity_mat[j][i]

        return similarity_mat

        

        for ref1 in reference_lst:
            vec1 = vectors[ref1]
            word_similarity = []  # a row vector which stores the similarity relationship for the current word(ref1). 
            for ref2 in reference_lst:
                vec2 = vectors[ref2]   
                similarity = 1 - spatial.distance.cosine(vec1, vec2) 
                word_similarity.append(similarity)      # add the similarity between ith word(ref1) and jth word(ref2) to the similarity row.
            similarity_mat.append(word_similarity)
        return similarity_mat
        

    num_reference_words = 10000
    reference_set = set() # a hashset to fast check if a word is in the reference word list to compute relationship similarity.
    reference_lst = []   # this list is used as an ordered container to store those reference words to compute relationship similarity.
    file_name = "XXX.txt"  # a word list containing words that the similarity matrix build on. 
    with open(file_name,"r") as reader:
        lines = reader.readlines()
        while len(reference_set) < num_reference_words:
            try:
               word = lines.pop(0).split(" ")[0] 
            except:
                pass
            else:
                if word.isalpha():
                    reference_set.add(word)
                    reference_lst.append(word)
    print ("number of reference words", len(reference_set))    

    vector_file = "YYY.txt"   # vector file 1
    vectors = load_vector(vector_file, reference_set)
    similarity_mat1 = compute_similarity_matrix(vectors, reference_lst)

    vector_file = "ZZZ.txt"  # vector file 2
    vectors = load_vector(vector_file, reference_set)
    similarity_mat2 = compute_similarity_matrix(vectors, reference_lst)

    diff = []
    for vec1, vec2 in zip(similarity_mat1, similarity_mat2):
        diff.append(np.absolute(vec1-vec2).sum())
    diff_arr = np.array(diff) # absolute difference between two word relationship vector.

    z_score = (diff_arr - diff_arr.mean())/diff_arr.std()   # compute z-score

    ans = {}
    for word, z in zip(reference_lst, z_score):
        ans[word] = z

    return ans


def main():
    ans = build_similarity_matrix()
    print ans

if __name__ == "__main__" : main()


