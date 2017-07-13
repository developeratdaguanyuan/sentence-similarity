import tensorflow as tf
import reader

from models import lstm
from models import bilstm

#from models import rnn
#from models import lstm
#from models import birnn
#from models import lstm_magic
#from models import decomposable_attention
#from models import intra_sentence_decomposable_attention
#from models import enhanced_lstm

data_dir = "../data"

def main(_):
    data = reader.build_data(data_dir)
    train_data = data['train_data']
    valid_data = data['valid_data']
    word_embedding = data['word_embedding']

    train_data_producer = reader.DataProducer(train_data)
    valid_data_producer = reader.DataProducer(valid_data, False)


    # BiLSTM
    graph = bilstm.BiLSTM(vocab_size=len(word_embedding),
                          class_size=1,
                          word_vectors=word_embedding)
    graph.train(train_data_producer, valid_data_producer, 10)

'''
    # LSTM
    graph = lstm.LSTM(vocab_size=len(word_embedding),
                      class_size=1,
                      word_vectors=word_embedding)
    graph.train(train_data_producer, valid_data_producer, 10)
'''

"""
  # intra_sentence_decomposable_attention
  graph = enhanced_lstm.EnhancedLSTMModel(vocab_size=vocabulary_size,
                                          class_size=1,
                                          word_vectors=word_vectors)
  graph.train(train_data_producer, valid_data_producer, 200)
"""

"""
  # intra_sentence_decomposable_attention
  graph = intra_sentence_decomposable_attention.IntraSentenceDecomposableAttentionModel(vocab_size=vocabulary_size,
                                                                                        class_size=1,
                                                                                        word_vectors=word_vectors)
  graph.train(train_data_producer, valid_data_producer, 200)
"""

"""
  # decomposable_attention
  graph = decomposable_attention.DecomposableAttentionModel(vocab_size=vocabulary_size,
                                                            class_size=1,
                                                            word_vectors=word_vectors)
  graph.train(train_data_producer, valid_data_producer, 200)
"""

"""
  # intra_sentence_decomposable_attention
  graph = intra_sentence_decomposable_attention.IntraSentenceDecomposableAttentionModel(vocab_size=vocabulary_size,
                                                                                        class_size=1,
                                                                                        word_vectors=word_vectors)
  graph.train(train_data_producer, valid_data_producer, 200)
"""

'''
  # LSTM_magic
  config = lstm_magic.ModelConfig(embedding_size, hidden_size, batch_size, learning_rate, vocabulary_size, 2)
  graph = lstm_magic.LSTM_magic(config, word_vectors, True)
  graph.train(train_data_producer, valid_data_producer, 200)
'''

'''
  # LSTM
  config = lstm.ModelConfig(embedding_size, hidden_size, batch_size, learning_rate, vocabulary_size, 2)
  graph = lstm.LSTM(config, word_vectors, False)
  graph.train(train_data_producer, valid_data_producer, 200)
'''

'''
  # BiRNN
  config = birnn.ModelConfig(hidden_size, batch_size, learning_rate, vocabulary_size, 2)
  graph = birnn.BiRNN(config, True)
  graph.train(train_data_producer, valid_data_producer, word_vectors, 100)
'''

'''
  # RNN
  config = rnn.ModelConfig(hidden_size, batch_size, learning_rate, vocabulary_size, 2)
  graph = rnn.RNN(config, True)
  graph.train(train_data_producer, valid_data_producer, word_vectors, 100)
'''

if __name__ == "__main__":
    tf.app.run()
