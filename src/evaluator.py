import tensorflow as tf
import reader

from models import lstm
from models import gru
from models import bigru
from models import bilstm
from models import bigru2layers

data_dir = '../data'

def main(_):
    # Data Producer
    data = reader.build_data(data_dir)
    test_data = data['test_data']
    word_embedding = data['word_embedding']
    data_producer = reader.DataProducer(test_data, False)

    # GRU
    model_path = '../save_models/gru_6328'
    graph = gru.GRU(vocab_size=len(word_embedding),
                    class_size=1,
                    word_vectors=word_embedding,
                    batch=100,
                    training=False)
    print graph.evaluate(data_producer, model_path)

'''
    # BiGRU2Layers
    model_path = '../save_models/bigru2layers_6328'
    graph = bigru2layers.BiGRU2Layers(vocab_size=len(word_embedding),
                                      class_size=1,
                                      word_vectors=word_embedding,
                                      batch=100,
                                      training=False)
    print graph.evaluate(data_producer, model_path)
'''

'''
    # BiLSTM
    model_path = '../save_models/bilstm_9492'
    graph = bilstm.BiLSTM(vocab_size=len(word_embedding),
                          class_size=1,
                          word_vectors=word_embedding,
                          batch=100,
                          training=False)
    print graph.evaluate(data_producer, model_path)
'''

'''
    # LSTM
    model_path = '../save_models/lstm_9492'
    graph = lstm.LSTM(vocab_size=len(word_embedding),
                      class_size=1,
                      word_vectors=word_embedding,
                      batch=100,
                      training=False)
    print graph.evaluate(data_producer, model_path)
'''

'''
    # BiGRU
    model_path = '../save_models/bigru_6328'
    graph = bigru.BiGRU(vocab_size=len(word_embedding),
                        class_size=1,
                        word_vectors=word_embedding,
                        batch=100,
                        training=False)
    print graph.evaluate(data_producer, model_path)
'''


if __name__ == "__main__":
    tf.app.run()

