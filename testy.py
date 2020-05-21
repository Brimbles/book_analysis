import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""

tokenized_text=sent_tokenize(text)
# print(tokenized_text)

tokenized_word=word_tokenize(text)
# print(tokenized_word)

fdist = FreqDist(tokenized_word)
print(fdist)

# fdist.most_common(2)

fdist.plot(30,cumulative=False)
plt.show()