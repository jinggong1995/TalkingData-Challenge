import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from preprocess import read_gz, get_index
import os
import pandas as pd
import gc


'''
get_variable done

regularizer done

loss_function done

train

dropout train test done
'''
class model:

	def __init__(self, feature_size, layers, field_size, embedding_size=100,
				 dropout_fm=1, dropout_prob=0.5, use_fm=True,
				 use_deep=True, l2_regularization_scale=0.0, lr=0.0005,
				 batch_size=256, loss_type='log', eval_metric=roc_auc_score,
				 optimizer='adam', initializer='xaiver'
				 ):
		'''
		:param feature_size: scalar: feature_size (one-hot feature vector length)
		:param layers:  list of scalars: layer dimensions for each layer in Deep part
		:param field_size:  scalar: number of features (input size)
		:param embedding_size: scalar: feature embedding size
							  (PS. embedding matrix size(feature_size,  embedding_size))
		:param dropout_fm: drop out rate for the FM part (first second order)
		:param dropout_prob: drop out rate for the Deep part (higher order)
		:param use_fm: use FM or not
		:param use_deep: use Deep NN or not
		:param l2_regularization_scale:  l2 regularization rate for the network
		:param epoch:
		:param lr:   learning rate
		:param batch_size:
		:param loss_type:  loss function types
		:param eval_metric:
		:param optimizer:  optimizer type
		:param initialier:
		'''

		self.feature_size = feature_size  # denote as M, size of the feature dictionary
		self.field_size = field_size  # denote as F, size of the feature fields
		self.embedding_size = embedding_size  # denote as K, size of the feature embedding

		self.dropout_fm = dropout_fm
		self.layer_dimensions = layers
		self.dropout_prob = dropout_prob
		self.use_fm = use_fm
		self.use_deep = use_deep
		self.l2_reg = l2_regularization_scale

		self.batch_size = batch_size

		if initializer == 'xaiver':
			self.initializer = tf.contrib.layers.xavier_initializer()
		else:
			self.initializer = None

		self.learning_rate = lr
		self.optimizer = optimizer

		self.loss_type = loss_type
		self.eval_metric = eval_metric
		self.train_result, self.valid_result = [], []

		self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)

		self.initial_weights()


	def initial_weights(self):
		'''
		:return: ALL weights in DeepFM model (1st, 2nd order and Neural Nets)
		'''
		self.weights = dict()

		regularizer = self.regularizer

		# embeddings
		self.weights['feat_embeddings'] = tf.get_variable(initializer=self.initializer,
														  name='feature_embeddings',
														  regularizer=regularizer,
														   shape=[self.feature_size,
																  self.embedding_size])

		self.weights['feat_bias'] = tf.get_variable(name='feature_bias',
													initializer=self.initializer,
													regularizer=regularizer,
													shape=[self.field_size, 1])

		# deep neural nets
		# do not need to

		# concat projection
		# final concat projection layer

		if self.use_fm and self.use_deep:
			input_size = self.field_size + self.embedding_size + self.layer_dimensions[-1]
		elif self.use_fm:
			input_size = self.field_size + self.embedding_size
		elif self.use_deep:
			input_size = self.layer_dimensions[-1]
		glorot = np.sqrt(2.0 / (input_size + 1))
		self.weights["concat_projection"] = tf.get_variable(name="concat_projection",
															shape=(input_size, 1),
															dtype=np.float32,
															initializer=self.initializer,
															regularizer=regularizer)  # layers[i-1]*layers[i]

		self.weights["concat_bias"] = tf.get_variable(name="concat_bias",
													  shape=(1, 1),
													  dtype=np.float32,
													  initializer=self.initializer,
													  regularizer=regularizer)

	def net(self, feat_index, feat_value, is_training):
		'''

		:param feat_index:  shape=[Batch_size, Field_size], values=[0, feat_size-1]
		:param feat_value:  shape=[Batch_size, Field_size]
		:param is_training:
		:return:
		'''

		if is_training:
			dropout_rate = self.dropout_prob
			dropout_fm = self.dropout_fm
		else:
			dropout_rate = 1.0
			dropout_fm = 1.0

		# model

		embeddings = tf.nn.embedding_lookup(self.weights['feat_embeddings'],
												 feat_index)
		feat_value = tf.reshape(feat_value, shape=[-1, self.field_size, 1])
		embeddings = tf.multiply(embeddings, feat_value)

		''' -----first order----- '''
		first_order = tf.nn.embedding_lookup(self.weights['feat_bias'], feat_index)
		first_order = tf.reduce_sum(tf.multiply(first_order, feat_value), 2)
		first_order = tf.nn.dropout(first_order, keep_prob=dropout_fm)

		''' -----second order----- '''
		# sum and square
		sum_square = tf.reduce_sum(embeddings, 1)
		sum_square = tf.square(sum_square)

		# square and sum
		square_sum = tf.square(embeddings)
		square_sum = tf.reduce_sum(square_sum, 1)

		# put together
		second_order = 0.5*tf.subtract(sum_square, square_sum)
		second_order = tf.nn.dropout(second_order, keep_prob=dropout_fm)

		''' -----deep neural net component----- '''
		deep_input = tf.reshape(embeddings, shape=[-1, self.field_size*self.embedding_size])
		deep_input = tf.nn.dropout(deep_input, keep_prob=dropout_rate)

		for ii in range(len(self.layer_dimensions)):

			deep_input = tf.layers.dense(deep_input, units=self.layer_dimensions[ii],
								  activation=tf.nn.relu, name='dense_' + str(ii),
								kernel_initializer=self.initializer, bias_initializer=self.initializer,
								kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)

			deep_input = tf.layers.batch_normalization(deep_input, name='bn_' + str(ii))

			deep_input = tf.nn.dropout(deep_input, keep_prob=dropout_rate)

		concat_input = tf.concat([first_order, second_order, deep_input], axis=1)
		out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

		return out


	def losses(self, label, out, weight):

		# loss
		if self.loss_type == "logloss":
			out = tf.nn.sigmoid(out)
			# put weight on cross entropy to compensate dataset unbalance
			loss = tf.losses.log_loss(label, out, weight=weight)
		else:
			loss = tf.nn.l2_loss(tf.subtract(label, out))
		
		# l2 regularization on weights
		if self.l2_reg > 0:
			loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

		return loss
  


	def get_batch(self, filename, batch_size):
		'''
		:return: a batch of data(tensor): shape=[batch_size, feature_length]
				 data label: shape=[batch_size, 1] (0, 1)

		'''

		'''data input pipeline'''

		filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

		reader = tf.TextLineReader(skip_header_lines=1)
		key, value = reader.read(filename_queue)

		record_defaults = [tf.constant([0], dtype=tf.float32),
						   tf.constant([0], dtype=tf.float32),  # Column 1
						   tf.constant([0], dtype=tf.float32),  # Column 2
						   tf.constant([0], dtype=tf.float32),  # Column 3
						   tf.constant([' '], dtype=tf.string),  # Column 4

						   tf.constant([0], dtype=tf.float32),  # Column 5
						   tf.constant([0], dtype=tf.float32),  # Column 6
						   tf.constant([0], dtype=tf.float32),  # Column 7
						   tf.constant([0], dtype=tf.float32), # Column 8
						   tf.constant([0], dtype=tf.float32),# Column 9
						   tf.constant([0], dtype=tf.float32), # Column 10
						   tf.constant([0], dtype=tf.float32), # Column 11
						   tf.constant([0], dtype=tf.float32), # Column 12
						   tf.constant([0], dtype=tf.float32), # Column 13
						   tf.constant([0], dtype=tf.float32), # Column 14
						   tf.constant([0], dtype=tf.float32), # Column 15
						   tf.constant([0], dtype=tf.float32), # Column 16
						   tf.constant([0], dtype=tf.float32), # Column 17
						   tf.constant([0], dtype=tf.float32), # Column 18(del)
						   tf.constant([0], dtype=tf.float32), # Column 19
						   tf.constant([0], dtype=tf.float32), # Column 20
						   tf.constant([0], dtype=tf.float32), # Column 21
						   tf.constant([0], dtype=tf.float32), # Column 22
						   tf.constant([0], dtype=tf.float32), # Column 23
						   tf.constant([0], dtype=tf.float32), # Column 24
						   tf.constant([0], dtype=tf.float32), # Column 25
						   tf.constant([0], dtype=tf.float32), # Column 26
						   tf.constant([0], dtype=tf.float32), # Column 27
						   tf.constant([0], dtype=tf.float32)] # Column 28

		col1, col2,col3,col4, col5, col6,col7, col8, col9, col10, col11, col12, col13, col14, col15, col16,col17, col18, col19, col20, col21, col22, col23, col24, col25,col26, col27, col28, col29 = tf.decode_csv(value, record_defaults=record_defaults)

		features = tf.stack([col2,col3, col6,col7, col9, col10, col11, col12, col13, col14, col15, col16,col17, col18, col20, col21, col22, col23, col24, col25, col26, col27, col28,col29])

		min_after_dequeue = 1000
		capacity = min_after_dequeue + 3 * batch_size

		example_batch, label_batch = tf.train.shuffle_batch(
			[features, col8], num_threads=32, batch_size=batch_size, capacity=capacity,
			min_after_dequeue=min_after_dequeue)

		return example_batch, label_batch


	def train(self, epochs, saving_dir, filewriter_path, batch_size):

		''' tensorboard parameters '''

		training_sample = 185000000


		''' data input '''

		#example_batch, label_batch = self.get_batch('new_train.csv', batch_size)


		'''build computational graph'''
		feat_index = tf.placeholder(tf.int32, shape=[None, None],
										 name="feat_index")  # None * F
		feat_value = tf.placeholder(tf.float32, shape=[None, None],
										 name="feat_value")  # None * F
		feat_label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1

		weight = tf.placeholder(tf.float32, shape=[None, 1], name="weight")  # None * 1

		out = self.net(feat_index, feat_value, is_training=True)

		loss = self.losses(feat_label, out=out, weight=weight)

		op = self.optim(loss)

		# accuracy
		
		pred = tf.greater(out, 0.0)
		correct = tf.equal(pred, tf.equal(feat_label, 1.0))
		acc = tf.reduce_mean(tf.cast(correct, tf.float32))

		# auc score
		probability = tf.sigmoid(out)
		auc, auc_op = tf.metrics.auc(feat_label, predictions=probability)

		# Now finally gather all the summaries and group them into one summary op.

		tf.summary.scalar('losses/Loss', loss)
		tf.summary.scalar('score/ACC', acc)
		tf.summary.scalar('score/AUC', auc)


		writer = tf.summary.FileWriter(filewriter_path)
		writer.add_graph(tf.get_default_graph())

		my_summary_op = tf.summary.merge_all()

		saver = tf.train.Saver()

		step = 0

		with tf.Session() as sess:

			init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
			sess.run(init_op)

			# Start populating the filename queue.
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			start = time.clock()

			for ii in range(epochs*int(training_sample/batch_size)):

				#print(str(ii),': ', time.clock())

				step += 1

				example = X[ii/epochs*batch_size:(ii+1)/epochs*batch_size, :]

				label = y[ii/epochs*batch_size:(ii+1)/epochs*batch_size]



				#example, label = sess.run([example_batch, label_batch])

				#print(1,': ', time.clock())


				index, value = get_index(example)

				#print(2,': ', time.clock())

				lbl = np.reshape(label, (batch_size, 1))

				my_weight = np.zeros_like(lbl)

				my_weight[my_weight==0] = 0.005
				my_weight[my_weight==1] = 0.995

				_, result, my_auc, my_acc = sess.run([op, probability, auc_op, acc], feed_dict={feat_index: index,
										feat_value: value,
										feat_label: lbl,
										weight: my_weight})
				#print(3,': ', time.clock())

				not_tf_auc = roc_auc_score(label, probability)

				if ii % 20 == 0:


					print('time used for 20 step: ', time.clock() - start)

					start = time.clock()

					summaries = sess.run(my_summary_op, feed_dict={feat_index: index,
																	 feat_value: value,
																	 feat_label: lbl,
																	 weight: my_weight})

					roc = sess.run(auc)

					print('training step: ', step, 'not_tf_auc: ', not_tf_auc, 'accuracy: ', my_acc)
					writer.add_summary(summaries, global_step=step)

				if ii % 10000 == 0:

					saver.save(sess, saving_dir, global_step=step)

			writer.close()

			coord.request_stop()
			coord.join(threads)

			saver.save(sess, saving_dir, global_step=step)


	def predict(self):

		pass

	def optim(self, loss):

		# optimizer

		if self.optimizer == "adagrad":
			optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
													   initial_accumulator_value=1e-8).minimize(loss)
		elif self.optimizer == "gd":
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
		elif self.optimizer == "momentum":
			optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
				loss)
		else:
			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
													epsilon=1e-8).minimize(loss)

		return optimizer

	def evaluate(self):

		pass


if __name__ == '__main__':

	# def __init__(self, feature_size, layers, field_size, embedding_size=10,
	#              dropout_fm=1, dropout_prob=0.5, use_fm=True,
	#              use_deep=True, l2_regularization_scale=0.0, epoch=5, lr=0.001,
	#              batch_size=256, loss_type='log', eval_metric=roc_auc_score,
	#              optimizer='adam', initializer='xaiver'
	#              ):

	nrows = 184903890
	nchunk = 5000000
	frm = nrows-100000000
	to = frm+nchunk
	train_df = pd.read_csv('new_train.csv', parse_dates=['click_time'], skiprows=range(1, frm),
	nrows=to-frm)
	train_df = train_df.sample(frac=1)
	gc.collect()

	cols = ['app', 'channel', 'device', 'ip',
		 'os', 'day', 'hour', 'ucount_channel_ip', 'mcount_app_ip_device_os', 
		 'ucount_hour_ip_day', 'ucount_app_ip', 
		 'ucount_os_ip_app', 'ucount_device_ip', 'ucount_channel_app', 'mcount_os_ip',
		 'ucount_os_app', 'count_t_ip', 'count_ip_app',
		  'count_ip_app_os', 'var_hour_ip_tchan', 'var_hour_ip_app_os', 
		  'var_day_ip_app_channel', 'mean_hour_ip_app_channel', 'nextClick', 'nextClick_shift']
	X = train_df[col].values
	y = train_df['is_attributed'].values
	del train_df
	gc.collect()

	feature_size = 975470

	field_size = 24

	embedding_size = 300

	DeepFM = model(feature_size=feature_size, layers=[3600, 1800, 128, 64, 1], field_size=field_size, embedding_size=embedding_size)

	epoch = 10

	batch_size = 500

	saving_addr = './log_v1'

	if not os.path.exists(saving_addr):
		os.mkdir(saving_addr)

	filewriter_path = './filewriter_v1'

	if not os.path.exists(filewriter_path):
		os.mkdir(filewriter_path)

	DeepFM.train(epoch, saving_dir=saving_addr, filewriter_path=filewriter_path, batch_size=batch_size)

