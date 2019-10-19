#!/usr/bin/python3
# coding: utf-8

import os
import sys
import pickle
import tensorflow as tf
import numpy as np
# import jieba

path_of_model = './model/checkpoints'
model_dir = "./model/"

class Predict():
    def __init__(self):
        # self.checkpoint_file = tf.train.latest_checkpoint(path_of_model)
        self.checkpoint_file = './model/checkpoints/model-4400'
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                # self.drop_keep_prob = graph.get_operation_by_name("drop_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                self.scores = graph.get_operation_by_name("output/scores").outputs[0]
                # softmax
                self.score = tf.nn.softmax(self.scores, 1)

        with open(os.path.join(model_dir, 'vocab_processor.pkl'), 'rb')as f:
            self.vocab_processor = pickle.load(f)

    def infer(self, sentences):
        # transfer to vector
        sentence_word = []
        for sentence in sentences:
            # sentence_word.append(' '.join(jieba.cut(sentence)))
            sentence_word.append(' '.join(list(sentence)))
        sentences_vectors = np.array(list(self.vocab_processor.fit_transform(sentence_word)))

        feed_dict = {
            self.input_x: sentences_vectors,
            # self.drop_keep_prob: 1.0
        }
        y, s = self.sess.run([self.predictions, self.score], feed_dict)
        # print(y, s)
        # self.sess.close()
        # 将数字转换为对应的意图
        # labels = [dicts[x] for x in y]
        s = [(x.argmax(), max(x)) for x in s]  # 负面：[1, 0], 正面：[0, 1]
        # print(s)
        return s

def predict(sentences):
    model = Predict()
    s = model.infer(sentences)
    print(s)
    return s

def main():
    if len(sys.argv) > 1:
        text = sys.argv[1]
        label = predict([text])
        print('`{}`模型预测的情感标签(1:正面；0:负面)及概率是：{}'.format(text, label))
    else:
        text1 = '内容不错，值得一读'
        text2 = '包装很好，看起来纸质也不错'
        text3 = "文章内容感觉不怎么好，好多地方都感觉是胡编乱扯的"
        predict([text1, text2, text3])

if __name__ == '__main__':
    main()