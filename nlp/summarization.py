# text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

text = """
Amad could yet play again this season for Manchester United.

The Ivory Coast international, one of our shining lights this term after winning three Player-of-the-Month awards since August, sustained an ankle injury in training ahead of the game at Tottenham Hotspur in mid-February.

Head coach Ruben Amorim admitted he feared that the no.16 would not be available again in 2024/25 but the boss delivered a more positive update on the 22-year-old, during his press conference to preview Sunday's Premier League encounter with Arsenal." 
"Even Amad, we'll see in the end of the last month," said Ruben in San Sebastian, after the 1-1 draw with Real Sociedad in the Europa League.

"I don't want to say anything to you but I have the hope to have Amad before, we'll see."

The Reds' season is due to end on 25 May, when we welcome Aston Villa to Old Trafford for a league encounter.

Amad is registered as part of the Europa League squad so would be eligible for that competition if United can remain on the road to Bilbao.

The Reds need to get past Real Sociedad on Thursday at Old Trafford, with tickets still available to purchase, in order to retain hopes of lifting the trophy on 21 May at San Mames.

The versatile forward, who has also impressed at wing-back, last played in the 2-1 win over Leicester City in the Emirates FA Cup. He has scored nine times this term, including a hat-trick against Southampton, and regularly been a bright attacking spark.

We wish Amad well in his bid to regain full fitness.
"""
from transformers import pipeline, AutoTokenizer

model = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model)

summarizer = pipeline("summarization", model)
result = summarizer(text)
print(result)

