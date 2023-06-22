import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# Note 1:
# About the similarities between "cat," "monkey," and "banana," it is important to note that these similarities are based on pre-trained word 
# vectors that capture semantic and contextual information from large text corpora. The word vectors aim to represent words in a multi-dimensional space, 
# where similar words are closer to each other. However, the interpretation of the similarity scores can be subjective, and the accuracy of the scores depends on the quality and size of the training data.

# As an example, let's consider the similarity between "car" and "bicycle." The word vectors may indicate a relatively low similarity score since cars
# and bicycles are different types of vehicles. However, it's possible that in a specific context, such as discussing transportation modes, the words "car" and
# "bicycle" could be considered more similar due to their shared functionality of personal transportation. The context and domain-specific knowledge 
# play an important role in determining the perceived similarity between words.


# Note 2:
# The 'en_core_web_md' model used in this example provides a good starting point for various NLP tasks. However, 
# the accuracy of the model's similarity scores can be further improved by fine-tuning or training on specific domain data. 
# It's important to note that the model's performance is influenced by the quality and size of the training data, as well as 
# the specific use case and context in which it is applied.