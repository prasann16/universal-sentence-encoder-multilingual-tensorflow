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

        return {"values" : np.inner(question_embeddings['outputs'], response_embeddings['outputs']).tolist()}

    def finalize(self):
        self.module = None

model = InferlessPythonModel()
model.initialize()
a = model.infer(
    {
        "questions": ["What is your age?"],
        "responses": ["I am 20 years old.", "good morning"],
        "response_contexts": ["I will be 21 next year.", "great day."],
    }
)
print(a)
