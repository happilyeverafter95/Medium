import wikipedia
import nltk 
from gensim.models import Word2Vec

chess = wikipedia.page("Chess").content

# split our document into sentences
sentences = nltk.sent_tokenize(chess) 

length = len(sentences)

stopwords = set(nltk.corpus.stopwords.words("english"))

for i in range(0, length): 
    
    # further tokenize our sentences
    temp = nltk.word_tokenize(sentences[i])
    
    # removing stop words, non-alpabetical tokens and converting to lower case 
    sentences[i] = [word.lower() for word in temp if word not in stopwords and word.isalpha()]    

# size refers to the desired dimensionality of vectors 
# window is upper bound in dynamic context window
model = Word2Vec(sentences, size=100, window=5)

# Exploring the model 

# Measures the similarity between words using cosine similarity 

model.similarity("rook", "knight") 

# Finds the top n most similar words 

model.similar_by_word("king",10)

# Find word using vector addition. Does opponent + checkmate = lose? 

opponent_checkmate = model.wv["opponent"] + model.wv["checkmate"] # add the vectors for king and checkmate
model.most_similar(positive = [opponent_checkmate], topn=10) # find most similar word to vector