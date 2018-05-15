import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import string
import matplotlib.pyplot as plt

from random import randint, choice

### to train on different data, just insert a path to a text file here:
corpus_path = '../corpora/CNCR_2017-18_corpus.txt'
###

# restore saved parameters and resume training?
restore=False
# if False then we just train from scratch

## call it whatever you want:
model_name = 'char_lstm'
model_path = 'model/'
summary_path = 'summary/'

print('Loading corpus...')
with open(corpus_path, encoding='latin1') as file:
    corpus = file.read()

#remove blank lines:
lines = corpus.split('\n')

# define legal characters:
legal_chars = string.ascii_lowercase + string.punctuation + string.whitespace + string.digits

def text_to_onehot(corpus, char_indices):
    """Takes a string and returns a ndarray of dimensions [len(string), num_chars],
    given a dict that maps characters to integer indices."""
    onehot = np.zeros((len(corpus), len(char_indices)))
    for x in range(len(corpus)):
        char = corpus[x]
        idx = char_indices[char]
        onehot[x,idx] = 1
    return onehot

def onehot_to_text(onehot, indices_char):
    """Takes an ndarray of softmax or onehot encoded text, and a dict that maps
    array indices to string characters, and returns the translated string."""
    text = []
    assert len(indices_char) == onehot.shape[1]
    for x in range(onehot.shape[0]):
        row = onehot[x,:]
        idx = np.argmax(row)
        char = indices_char[idx]
        text.append(char)
    return ''.join(text)

chars = sorted(legal_chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

num_chars = len(chars)

print('Processing corpus...')
newlines = []
for x in lines:
    if x != '' and x[0] != '!' and x[0] != '.': # remove empty lines and bot responses
        if '<@' not in x: # remove mentions
            legalx = ''.join([i for i in x.lower() if i in legal_chars])
            newlines.append(legalx.replace('&&newline', '\n'))
raw_corpus = '\n'.join(newlines)
corpus = text_to_onehot(raw_corpus, char_indices)

### corpus loaded, now we construct the batch generator:

def get_batch(corpus, batch_size, str_len, char_pred = True):
    # if char_pred = False, returns an array of random strings taken from the corpus
    # if char_pred = True, instead returns two arrays, one x and one y (shifted right by one) for character prediction
    corpus_size, num_chars = corpus.shape
    if not char_pred:
        batch = np.zeros((batch_size, str_len, num_chars))
        for b in range(batch_size):
            start_idx = randint(0, corpus_size - str_len - 1) # randint is end-inclusive
            end_idx = start_idx + str_len
            batch[b,:,:] = corpus[start_idx:end_idx,:]
        return batch
    else:
        xbatch = np.zeros((batch_size, str_len, num_chars))
        ybatch = np.zeros((batch_size, str_len, num_chars))
        for b in range(batch_size):
            start_x = randint(0, corpus_size - str_len - 2) # randint is end-inclusive
            end_x = start_x + str_len
            start_y, end_y = start_x + 1, end_x + 1
            xbatch[b,:,:] = corpus[start_x:end_x,:]
            ybatch[b,:,:] = corpus[start_y:end_y,:]
        return xbatch, ybatch


## build the network:
print('Constructing network...')

sess = tf.InteractiveSession()

lstm_size = 300
str_len = 50
batch_size = 200
learning_rate = 0.001

x = tf.placeholder(tf.float32, [None, None, num_chars], name='x')
y = tf.placeholder(tf.float32, [None, None, num_chars], name='y')

num_cells = 2

## lstm:
cells = [rnn.BasicLSTMCell(lstm_size) for _ in range(num_cells)]
multicell = rnn.MultiRNNCell(cells)
projection = rnn.OutputProjectionWrapper(multicell, num_chars)

# outputs for training:
rnn_outputs, final_state = tf.nn.dynamic_rnn(projection, x, dtype=tf.float32)

xe = tf.nn.softmax_cross_entropy_with_logits(logits=rnn_outputs, labels=y)
total_loss = tf.reduce_mean(xe)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

# outputs for sequential text generation:
seq_init = projection.zero_state(1, dtype=tf.float32)
seq_len = tf.placeholder(dtype=tf.int32, name='seq_len')
seq_output, seq_state = tf.nn.dynamic_rnn(projection, x, initial_state=seq_init, sequence_length=seq_len)

print('Initialising variables...')
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

if restore:
    saver.restore(sess, model_path)

def softmax(k):
    """Compute softmax values for each sets of scores in x."""
    e_k = np.exp(k - np.max(k))
    return e_k / e_k.sum(axis=0)

def generate_sequence(seed='\n', max_len=100, stop_at_newline=True, complete_sentence=False):
    """recursively generate a sequence by generating a prediction and feeding it to next time step.
      args:
    seed:: initial data to feed to network. has to be non-empty, but newline \n is a safe bet.
    max_len:: stop generating when we reach this many characters.
    stop_at_newline:: if True, stop generating when a newline \n is generated.
    complete_sentence:: if True, return the seed as well as the generated sequence.

    this function might need cleaning up, but it works ok"""

    seed_onehot = text_to_onehot(seed, char_indices)
    pred_chars = []
    states = []
    state = None
    for i in range(len(seed)):
        seed_in = seed_onehot[i,:].reshape(1, 1, -1)
        feed = {x: seed_in, seq_len: 1}
        if i > 0:
            feed[seq_init] = state
        out, state = sess.run([seq_output, seq_state], feed_dict=feed) # print seed
        # char_onehot = out[0,0,:]
        # char = onehot_to_text(char_onehot.reshape(1,-1), indices_char)
        if complete_sentence:
            pred_chars.append(seed[i])
        states.append(state)

    # process last state before generating sequence:
    char_logits = out[0, 0, :]
    char_softmax = softmax(char_logits)
    char_range = np.arange(num_chars)
    char_choice = np.random.choice(char_range, p=char_softmax)
    char_onehot = np.eye(num_chars)[char_choice, :]  # neat trick
    char = onehot_to_text(char_onehot.reshape(1, -1), indices_char)
    pred_chars.append(char)

    done = False
    i = 0
    while not done:
        feed = {x: char_onehot.reshape(1, 1, -1), seq_init: state, seq_len: 1}
        out, state = sess.run([seq_output, seq_state], feed_dict=feed) # print prediction
        char_logits = out[0, 0, :]
        char_softmax = softmax(char_logits)
        char_range = np.arange(num_chars)
        char_choice = np.random.choice(char_range, p=char_softmax)
        char_onehot = np.eye(num_chars)[char_choice,:] # neat trick
        char = onehot_to_text(char_onehot.reshape(1,-1), indices_char)
        pred_chars.append(char)
        states.append(state)

        i += 1
        if i > max_len or ((stop_at_newline) and char == '\n'):
            done = True
            pred_chars = pred_chars[:-1]
    sequence = ''.join(pred_chars)
    if len(sequence) > 0 and sequence[0] == '\n':
        sequence = sequence[1:]
    return sequence


# generate a test sequence:
generate_sequence(seed='\n', max_len=100, complete_sentence=True)

### now train the network:

num_epochs = 1000
update_interval = 100

ytexts = []
predtexts = []
gentexts = []
losses = []

print('Training network...')
batch_times = []
feed_times = []
e = 0

training_time = 1*60*60 # in seconds
done = False
start_time = time.time()

# train for an amount of time:
while not done:
    time_elapsed = time.time() - start_time

    batch_start = time.time()
    xbatch,ybatch = get_batch(corpus, batch_size, str_len, char_pred=True)
    batch_end = time.time()
    feed = {x: xbatch, y: ybatch}
    _, loss, pred = sess.run([train_step, total_loss, rnn_outputs], feed_dict=feed)
    feed_end = time.time()

    batch_times.append(batch_end - batch_start)
    feed_times.append(feed_end - batch_end)

    losses.append(loss)
    # see what we're producing:
    if e % update_interval == 0:
        print('  Time elapsed: %d secs. Epoch %d: Training loss %.6f' % (time_elapsed, e, loss))
        ytext = onehot_to_text(ybatch[0], indices_char)
        predtext = onehot_to_text(np.array(pred)[0,:,:], indices_char)
        gentext = generate_sequence()
        print('  desired sentence:')
        print('        %s' % ytext.replace('\n', '\\n'))
        print('  output sentence:')
        print('        %s' % predtext.replace('\n', '\\n'))
        ytexts.append(ytext)
        predtexts.append(predtext)
        print('  generated sentence:')
        print('        %s' % gentext)
        gentexts.append(gentext)
        print('Time spent generating batches: %.6f' % np.sum(batch_times))
        print('Time spent training network: %.6f' % np.sum(feed_times))

        batch_times = []
        feed_times = []

    e += 1
    if time_elapsed > training_time:
        print('Training complete.')
        done = True

plt.plot(losses)


# generate a series of test sequences, each seeded by the last:
sequences=['\n']
for i in range(10):
    sequences.append(generate_sequence(seed=sequences[-1], max_len=100))
print('\n'.join(sequences))

# save weights:
save_path = saver.save(sess, model_path)
print("TF model variables for %s saved in path: %s" % (model_name, save_path))