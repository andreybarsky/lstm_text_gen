from corpus_loader import load_corpus, get_counts, get_orders
import numpy as np
import math
import tensorflow as tf

path = '../corpora/very_official_discord_server_corpus.txt'

words = load_corpus(path, level='words')

counts = get_counts(words) # dict of words by their count

vocab_size = len(counts)

# words in descending count order:
desc_counts = list(reversed(sorted(counts.values())))
desc_words = get_orders(counts)
word_ids = np.arange(len(desc_words)) # vector of word index values
word_dict = dict(zip(desc_words, word_ids))
id_dict = dict(zip(word_ids, desc_words))
unk_id = len(desc_words) # ID of unknown words, which don't exist in our corpus

int_words = np.array([(word_dict[w] if w in word_dict else unk_id) for w in words])

def skipgram_batch(int_words, batch_size=100, w=5, dict_size=None):
    """Takes list of word tokens as integers and returns a matrix of one-hot encoded skipgrams"""
    
    # if dict_size is not given we perform an extra computation to infer it:    
    if dict_size is None:
        dict_size = len(set(int_words))

    corpus_size = len(int_words)

    # assume that batch_size is a multiple of w*2:
    num_contexts = w * 2
    # num_targets = batch_size // num_contexts # determine number of target words in the batch

    mindex = w
    maxdex = len(int_words)-w

    # corpus indices of target words:
    target_idxs = np.random.randint(mindex, maxdex, (batch_size, 1)).reshape(1,-1)

    w_offsets = np.concatenate((np.arange(-w,0), np.arange(1,w+1))).reshape(-1,1)

    # corpus indices of context words:
    context_idxs = (w_offsets + target_idxs)
    # repeat targets so that each one matches a context:
    # target_idxs = np.repeat(target_idxs, num_contexts)

    context_tokens = int_words[context_idxs]

    context_nhot = np.zeros((batch_size, dict_size))
    for i in range(batch_size):
       context_nhot[i,context_tokens[:,i]] = 1.0 # encode context words as n-hot matrix

    # token ids of target words:
    target_tokens = int_words[target_idxs]

    return target_tokens, context_nhot

### now construct the network:

batch_size = 100
embedding_size = 500
num_sampled = 64 # number of nce pairs

x = tf.placeholder(tf.int32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None, vocab_size])

# one hot and squeeze:
# x_onehot = tf.reshape(tf.one_hot(x, vocab_size), (-1, vocab_size))

embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
# b1 = tf.Variable(tf.random_normal([vocab_size, embedding_size]))

embed = tf.nn.embedding_lookup(embeddings, x)

# h1 = tf.matmul(x_onehot, embedding) # + b1
# output = tf.matmul(h1, w2) + b2
# softmax = tf.nn.softmax(output)

w2 = tf.Variable(tf.random_normal([embedding_size, vocab_size]))
b2 = tf.Variable(tf.random_normal([1, vocab_size]))

nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
    labels=y, inputs=embed, num_sampled=num_sampled,
    num_classes=vocab_size))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(nce_loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# train network:

feed = skipgram_batch(int_words, batch_size)
optimizer.eval(feed)