import tensorflow as tf
import logging


class GRU(object):
  def __init__(self, vocab_size, class_size, word_vectors,
               batch=100, embedding_dim=300, hidden_dim=300, learning_rate=1e-3,
               training=True):
    '''
    Build GRU Model
    :param vocab_size: vocabulary size
    :param class_size: class size
    :param word_vectors: word vectors for initilization
    :param batch: batch size
    :param embedding_dim:
    :param hidden_dim:
    :param learning_rate:
    :param training: true if do training, false if just do evaluation
    '''
    self.batch = batch

    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.lr = learning_rate
    self.vocab_sz = vocab_size
    self.class_sz = class_size

    # Placeholders
    # Sentence placeholders with shape (batch, time_steps)
    self.s1 = tf.placeholder(tf.int32, [self.batch, None])
    self.s2 = tf.placeholder(tf.int32, [self.batch, None])
    # Label placeholders with shape (batch, 1)
    self.y = tf.placeholder(tf.float32, [self.batch, self.class_sz])

    # Embedding layer
    # Tensor e{1, 2} have shape (batch, time_steps, embedding_dim)
    self.embeddings = tf.get_variable('embedding_matrix', dtype='float',
                                      initializer=tf.constant_initializer(word_vectors),
                                      shape=[self.vocab_sz, self.embedding_dim], trainable=False)
    e1 = tf.nn.embedding_lookup(self.embeddings, self.s1)
    e2 = tf.nn.embedding_lookup(self.embeddings, self.s2)

    # GRU
    z1, _ = self._apply_gru(e1, self.hidden_dim, 'encoder')
    z2, _ = self._apply_gru(e2, self.hidden_dim, 'encoder', True)
    last_z1 = tf.transpose(z1, perm=[1, 0, 2])[-1]
    last_z2 = tf.transpose(z2, perm=[1, 0, 2])[-1]
    z = tf.concat(1, [tf.abs(tf.subtract(last_z1, last_z2)), tf.multiply(last_z1, last_z2)])

    #if training:
    #  z = tf.nn.dropout(z, 0.85)

    # Softmax layer
    with tf.variable_scope('relu'):
      W_relu = tf.get_variable('W_relu', [self.hidden_dim * 2, self.hidden_dim])
      b_relu = tf.get_variable('b_relu', [self.hidden_dim], initializer=tf.constant_initializer(0.0))
      y_relu = tf.nn.relu(tf.matmul(z, W_relu) + b_relu)

    #if training:
    #  y_relu = tf.nn.dropout(y_relu, 0.85)

    with tf.variable_scope('sigmoid'):
      W_sigmoid = tf.get_variable('W_sigmoid', [self.hidden_dim, self.hidden_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
      b_sigmoid = tf.get_variable('b_sigmoid', [self.hidden_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
      y_sigmoid = tf.tanh(tf.matmul(y_relu, W_sigmoid) + b_sigmoid)

    with tf.variable_scope('softmax'):
      W_softmax = tf.get_variable('W_softmax', [self.hidden_dim, self.class_sz], initializer=tf.random_uniform_initializer(-0.1, 0.1))
      b_softmax = tf.get_variable('b_softmax', [self.class_sz], initializer=tf.random_uniform_initializer(-0.1, 0.1))

    z = tf.matmul(y_sigmoid, W_softmax) + b_softmax
    self.preds = tf.nn.sigmoid(z)

    # loss & optimize
    self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = z, targets = self.y, pos_weight = 1)) #0.360574285))
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


  def _apply_gru(self, input, unit_num, scope=None, reuse=False):
    '''
    GRU model
    :param input: word vectors of a sentence with shape (batch, length, embedding_dim)
    :param unit_num: number of unit in LSTM
    :param reuse: reuse layers
    :return:
    '''
    scope_name = scope or 'gru'
    with tf.variable_scope(scope_name, reuse=reuse):
      initializer = tf.random_uniform_initializer(-0.1, 0.1)
      cell = tf.nn.rnn_cell.GRUCell(unit_num)
      z, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)

    return z, final_state


  def train(self, train_data_producer, valid_data_producer, num_epoch):
    '''
    :param train_data_producer: train data producer
    :param valid_data_producer: valid data producer
    :param num_epoch: number of epoch
    :return:
    '''
    logging.basicConfig(filename='../logs/gru.log', filemode='w', level=logging.INFO)
    saver = tf.train.Saver()
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())

      train_loss, valid_loss = 0, float('inf')
      num_iteration = num_epoch * train_data_producer.size / self.batch
      for i in range(num_iteration):
        question_1, question_2, _, _, label = train_data_producer.next(self.batch)
        feed = {self.s1: question_1, self.s2: question_2, self.y: label}

        ret = self.sess.run([self.optimizer, self.loss], feed_dict=feed)
        train_loss += ret[1]

        if i > 0 and i % (train_data_producer.size / self.batch) == 0:
          # train info
          train_loss = train_loss / train_data_producer.size * self.batch
          logging.info("[train cross entropy] %5.3f", train_loss)
          train_loss = 0

          # valid info
          valid_loss_t = self._evaluate(valid_data_producer)
          logging.info("[valid cross entropy] %5.3f", valid_loss_t)

          # write model
          if valid_loss_t < valid_loss:
            valid_loss = valid_loss_t
            save_path = saver.save(self.sess, "../save_models/gru_" + str(i))
            logging.info("Model saved in file: %s", save_path)


  def _evaluate(self, data_producer):
    '''
    Do Evaluation during training
    :param data_producer: data producer
    :return: log loss
    '''
    loss_t = 0
    while True:
      data = data_producer.next(self.batch)
      if data is None:
        loss_t = loss_t / data_producer.size * self.batch
        break
      question_1, question_2, _, _, label = data
      feed = {self.s1: question_1, self.s2: question_2, self.y: label}
      ret = self.sess.run([self.loss], feed_dict=feed)
      loss_t += ret[0]

    return loss_t


  def evaluate(self, data_producer, model_path):
    '''
    Do Evaluation
    :param data_producer: data producer
    :param model_path: model path
    :return: log loss
    '''
    with tf.Session() as self.sess:
      self.sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(self.sess, model_path)
      return self._evaluate(data_producer)

    return None
