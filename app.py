import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

class InferlessPythonModel:
    def initialize(self):
        self.module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

    def infer(self, inputs):
        questions = inputs['questions']
        responses = inputs['responses']
        response_contexts = inputs['response_contexts']

        question_embeddings = self.module.signatures['question_encoder'](tf.constant(questions))
        response_embeddings = self.module.signatures['response_encoder'](input=tf.constant(responses), context=tf.constant(response_contexts))

        return {"values" : str(np.inner(question_embeddings['outputs'], response_embeddings['outputs']))}

    def finalize(self):
        self.module = None

