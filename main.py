from data_processor import *

# data
q1, q2, y = get_data(name='bq_corpus')
seg_q1 = [word_segmentation(q) for q in q1]
seg_q2 = [word_segmentation(q) for q in q2]

# doc_representation
# TODO


if __name__ == '__main__':
    print(q1[0], seg_q1[0], y[0])
    print(q1[1], seg_q1[1], y[0])
