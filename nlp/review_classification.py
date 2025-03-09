text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
result = classifier(text)
print(result)

text2 = "this was a somewhat lukewarm piece. I consider myself an sci-fi fan. But this movie hardly meet my standards for the category. It was a popcorn movie. fit for spending hour and a half chilling."
result2 = classifier(text2)
print(result2)

# 한글 가능 모델
# classifier = pipeline("sentiment-analysis", model="WhitePeak/bert-base-cased-Korean-sentiment")

text3 = "this movie gave me a dream. Anyone could be a movie director."
result3 = classifier(text3)
print(result3)

text4 = "this movie gave me a dream. Anyone could be a movie director. Even I can make this kind of film."
result4 = classifier(text4)
print(result4)

