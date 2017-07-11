import tensorflow as tf
import reader

from models import lstm

data_dir = '../data'
model_path = '../save_models/lstm_9492'

def main(_):
    # Data Producer
    data = reader.build_data(data_dir)
    test_data = data['test_data']
    word_embedding = data['word_embedding']
    data_producer = reader.DataProducer(test_data, False)

    # LSTM
    graph = lstm.LSTM(vocab_size=len(word_embedding),
                      class_size=1,
                      word_vectors=word_embedding,
                      batch=100,
                      training=False)
    print graph.evaluate(data_producer, model_path)
    
if __name__ == "__main__":
    tf.app.run()

