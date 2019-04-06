import tqdm
from collections import Counter
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import argparse
import os
import _pickle as pickle
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
import process
import spacy

def grab_data(num_samples):
	parent, reply = [], []
	for i,comment in enumerate(open('../processed_data/processed_parent.txt', 'r')):
		if i == num_samples:
			break
		parent.append(comment[:-1].split())
	for i,comment in enumerate(open('../processed_data/processed_reply.txt', 'r')):
		if i == num_samples:
			break
		reply.append(comment[:-1].split())
	return parent,reply

def one_hot(vec, num_features):
	toRet = np.zeros((vec.shape[0], vec.shape[1], num_features), dtype = np.int32)
	for i in range(len(vec)):
	    toRet[i, np.arange(len(vec[i])), vec[i]] = 1
	return toRet

def pad(vec, pad_token, size):
	return np.array([i+[pad_token]*(size-len(i)) for i in vec])

def load_data(parent_data, reply_data, lower, upper):
	parent, reply = parent_data[lower:upper], reply_data[lower:upper]
	enc_seq_len = [min(max_enc_time, len(comment)) for comment in parent]
	dec_seq_len = [min(max_dec_time, len(comment)) for comment in reply]
	max_enc_pad = max(enc_seq_len)
	max_dec_pad = max(dec_seq_len)
	enc_input = pad([[parent_w2i[word] if word in parent_w2i else parent_w2i["PAD"] for word in comment[:max_enc_pad]] for comment in parent], parent_w2i["PAD"], max_enc_pad)
	dec_input = pad([[reply_w2i["SOS"]]+[reply_w2i[word] if word in reply_w2i else reply_w2i["PAD"] for word in comment[:max_dec_pad-1]] for comment in reply], reply_w2i["PAD"], max_dec_pad)
	dec_target = one_hot(pad([[reply_w2i[word] if word in reply_w2i else reply_w2i["PAD"] for word in comment[:max_dec_pad-1]]+[reply_w2i["EOS"]] for comment in reply], reply_w2i["PAD"], max_dec_pad), dec_features)
	return enc_input, dec_input, dec_target, np.array(enc_seq_len), np.array(dec_seq_len)

def get_args():
	parser = argparse.ArgumentParser(description = "Reddit Chatbot")
	parser.add_argument("-e", "--epochs", type = int)
	parser.add_argument("-s", "--num_samples", type = int)
	parser.add_argument("-l", "--latent_dim", type = int)
	parser.add_argument("-d", "--embedding_dim", type = int)
	parser.add_argument("--batch_size", type = int)
	parser.add_argument("--num_layers", type = int)
	parser.add_argument("--beam_width", type = int)

	return parser.parse_args()

args = get_args()

num_samples = args.num_samples if args.num_samples else 500
latent_dim = args.latent_dim if args.latent_dim else 256
embedding_dim = args.embedding_dim if args.embedding_dim else 100
epochs = args.epochs if args.epochs else 100
batch_size = args.batch_size if args.batch_size else 100
num_layers = args.num_layers if args.num_layers else 1
beam_width = args.beam_width if args.beam_width else 3

print("Loading Data")
parent, reply = grab_data(num_samples)

print("Creating frequency dictionaries")
parent_freq_dict = Counter(word.lower() for comment in parent for word in comment)
reply_freq_dict = Counter(word.lower() for comment in reply for word in comment)

print("Creating vocabularies")
enc_features = min(30000, len(parent_freq_dict))
dec_features = min(30000, len(parent_freq_dict))

parent_vocab = ["PAD"]+[word[0] for word in parent_freq_dict.most_common(enc_features-1)]
reply_vocab = ["SOS", "PAD"]+[word[0] for word in reply_freq_dict.most_common(dec_features-3)] + ["EOS"]

max_enc_time = 50
max_dec_time = 50

print("Creating mapping dictionaries")
parent_w2i, reply_w2i = {parent_vocab[idx-1]:idx for idx in range(1, len(parent_vocab)+1)}, {reply_vocab[idx-1]:idx for idx in range(1, len(reply_vocab)+1)}
parent_i2w, reply_i2w = {integer:word for word,integer in parent_w2i.items()}, {integer:word for word,integer in reply_w2i.items()}

def get_placeholders():
        enc_input = tf.placeholder(shape = [None, None], dtype = tf.int32, name = 'enc_input')
        enc_seq_len = tf.placeholder(shape = [None], dtype = tf.int32, name = 'enc_seq_len')
        dec_input = tf.placeholder(shape = [None, None], dtype = tf.int32, name = 'dec_input')
        dec_seq_len = tf.placeholder(shape = [None], dtype = tf.int32, name = 'dec_seq_len')
        dec_target = tf.placeholder(shape = [None, None, dec_features], dtype = tf.int32, name = 'dec_target')

        return enc_input, dec_input, dec_target, enc_seq_len, dec_seq_len

def gen_attention_cell(mode, enc_seq_len, attention_states, init_dec_state, beam_width, batch_size):
    if mode == 'infer':
        attention_states = tf.contrib.seq2seq.tile_batch(attention_states, beam_width)
        init_dec_state = tf.contrib.seq2seq.tile_batch(init_dec_state, beam_width)
        enc_seq_len = tf.contrib.seq2seq.tile_batch(enc_seq_len, beam_width)
        batch_size = batch_size * beam_width

    attention_mech = tf.contrib.seq2seq.LuongAttention(latent_dim*2, attention_states, memory_sequence_length = enc_seq_len, name = 'LuongAttention')
    decoder_lstm = tf.nn.rnn_cell.BasicLSTMCell(latent_dim*2, name = 'decoder_lstm')
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_lstm, attention_mech, name = 'attention_cell')

    return attention_cell, attention_cell.zero_state(dtype = tf.float32, batch_size = batch_size).clone(cell_state = init_dec_state)

def Decoder(mode, dec_emb_input, enc_seq_len, dec_seq_len, attention_states, final_enc_state, beam_width, batch_size, dec_emb_matrix):
    with tf.variable_scope("Decoder") as decoder_scope:
        attention_cell, init_dec_state = gen_attention_cell(mode, enc_seq_len, attention_states, final_enc_state, beam_width, batch_size)
        projection_layer = Dense(dec_features, use_bias = False, name = 'projection_layer')

        if mode == 'train':
            helper = tf.contrib.seq2seq.TrainingHelper(dec_emb_input, dec_seq_len, name = 'helper')
            decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, init_dec_state, projection_layer)
        else:
            start_tokens = tf.tile(np.array([reply_w2i["SOS"]], dtype = np.int32), [batch_size])
            end_token = reply_w2i["EOS"]
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(attention_cell, dec_emb_matrix, start_tokens, end_token, init_dec_state, beam_width, projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations = max_dec_time, scope = decoder_scope)

    return outputs

def construct_graph(mode, placeholders, batch_size):
	enc_input, dec_input, dec_target, enc_seq_len, dec_seq_len = placeholders

	enc_emb_matrix = tf.get_variable("enc_embedding_matrix", [enc_features, embedding_dim])
	dec_emb_matrix = tf.get_variable("dec_embedding_matrix", [dec_features, embedding_dim])
	enc_emb_input = tf.nn.embedding_lookup(enc_emb_matrix, enc_input)
	dec_emb_input = tf.nn.embedding_lookup(dec_emb_matrix, dec_input)

	with tf.name_scope("Encoder"):
		fw_cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(latent_dim), output_keep_prob = 0.8) for i in range(num_layers)])
		bw_cells = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(latent_dim), output_keep_prob = 0.8) for i in range(num_layers)])

		enc_outputs, enc_states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, enc_emb_input, sequence_length = enc_seq_len, time_major = False, dtype = tf.float32)
		enc_outputs = tf.concat(enc_outputs, -1)
		enc_c = tf.concat([enc_states[0][-1][0], enc_states[1][-1][0]], -1)
		enc_h = tf.concat([enc_states[0][-1][1], enc_states[1][-1][1]], -1)
		enc_states = tf.contrib.rnn.LSTMStateTuple(c = enc_c, h = enc_h)

	loss, optimizer = None, None
	outputs = Decoder(mode, dec_emb_input, enc_seq_len, dec_seq_len, enc_outputs, enc_states, beam_width, batch_size, dec_emb_matrix)

	if mode == 'train':
		outputs = outputs.rnn_output
		print(outputs)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = dec_target, logits = outputs))
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
	else:
		outputs = tf.transpose(outputs.predicted_ids, [0,2,1], name = 'inf_output')

	return outputs, loss, optimizer

def train_model(train_sess, train_saver, placeholders, loss, optimizer, output):
	enc_input, dec_input, dec_target, enc_seq_len, dec_seq_len = placeholders
	for epoch in range(epochs):
		i = 0
		cost = 0
		for i in tqdm.tqdm(range(0,num_samples, batch_size)):
			start, end = i, i+batch_size
			epoch_enc_input, epoch_dec_input, epoch_dec_target, epoch_enc_seq_len, epoch_dec_seq_len = load_data(parent, reply, start, end)
			# print(dec_target)
			# print(epoch_dec_target.shape)
			# print(max(epoch_dec_seq_len))
			# exit()
			#epoch_enc_seq_len, epoch_dec_seq_len = np.array([max_enc_time]*batch_size), np.array([max_dec_time]*batch_size)
			_, c = train_sess.run([optimizer, loss], feed_dict = {enc_input:epoch_enc_input, enc_seq_len: epoch_enc_seq_len, dec_input:epoch_dec_input,
			    dec_seq_len:epoch_dec_seq_len, dec_target:epoch_dec_target})
			cost += c
		# if not epoch%10:
		# 	if not os.path.exists('../beam_train'):
		# 		os.makedirs("../beam_train")
		# 	os.makedirs("../beam_train/model"+str(round(cost,2)))
		# 	train_saver.save(train_sess, "beam_train/model"+str(round(cost, 2))+"/beam_model")
		print("Finished epoch", epoch+1, " Loss:", cost,"\n\n")
	train_saver.save(train_sess, "beam_train/beam_model")
	pickle.dump([parent_w2i, reply_w2i, max_enc_time], open("beam_infer/params.pickle", 'wb'))

tf.reset_default_graph()
train_graph = tf.Graph()
infer_graph = tf.Graph()

with train_graph.as_default():
    train_placeholders = get_placeholders()
    train_output, loss, optimizer = construct_graph('train', train_placeholders, batch_size)
    initializer = tf.global_variables_initializer()
    train_saver = tf.train.Saver()

with infer_graph.as_default():
    infer_placeholders = get_placeholders()
    infer_output, _, _ = construct_graph('infer', infer_placeholders, 2)
    infer_saver = tf.train.Saver()

train_sess = tf.Session(graph = train_graph)
infer_sess = tf.Session(graph = infer_graph)

train_sess.run(initializer)
train_model(train_sess, train_saver, train_placeholders, loss, optimizer, train_output)

infer_saver.restore(infer_sess, 'beam_train/beam_model')
infer_saver.save(infer_sess, "beam_infer/beam_model")

with infer_graph.as_default():
	while True:
		enc_input = infer_graph.get_tensor_by_name("enc_input:0")
		enc_seq_len = infer_graph.get_tensor_by_name("enc_seq_len:0")

		inp = input("Enter a sentence:")
		if inp == "quit()":
			break
		nlp = spacy.load('en')
		inp = process.tokenize(inp, nlp, 'p')
		inp = np.array([[parent_w2i[word] if word in parent_w2i else parent_w2i["PAD"] for word in inp]]).reshape((1,-1))
		# inp = pad(inp, parent_w2i["PAD"], max_enc_time).reshape((1,-1))
		input_seq = np.concatenate([inp, inp])
		out = infer_sess.run([infer_output], feed_dict = {enc_input: input_seq, enc_seq_len:np.array([len(inp[0])]*2)})[0][0]
		print([" ".join([reply_i2w[idx] for idx in sentence]) for sentence in out])
		print("\n\n\n")
