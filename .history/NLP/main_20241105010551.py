from text_process.text_processor import TextPreprocessor

text = "London was the old capital of United Kingdom which has the most population in the world and I'll be go there someday with Hung's girlfriend who is in the U.S now."

preprocessor = TextPreprocessor()
preprocessor.set_text(text)  # Set the text using the set_text method

# Tokenization using different methods
tokens_1 = preprocessor.tokenize(method='whitespace')
print("Kết quả tách từ whitespace:", tokens_1)
print('-' * 60)

tokens_2 = preprocessor.tokenize(method='word_punct')
print("Kết quả tách từ word_punct:", tokens_2)
print('-' * 60)

tokens_3 = preprocessor.tokenize(method='treebank')
print("Kết quả tách từ treebank:", tokens_3)
print('-' * 60)

stemmed_tokens = preprocessor.stemming()  # No need to pass tokens_1
print("Stemming:", stemmed_tokens)
print('-' * 60)
lemmatized_tokens = preprocessor.lemmatization()  # No need to pass tokens_1
print("Lemmatization:", lemmatized_tokens)