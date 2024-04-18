from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from config import config

mpdel_path = "./nlp_model/" + config.default.sentiment_analysis_model_name
tokenizer = AutoTokenizer.from_pretrained(mpdel_path)
model = AutoModelForSequenceClassification.from_pretrained(mpdel_path)
nlp = pipeline('sentiment-analysis', model=model,tokenizer=tokenizer)

def sentiment_analysis(text: str) -> int:
    nlp_output = nlp(text)
    print(nlp_output)
    #star = int(nlp_output[0][0]["label"].lstrip("star "))
    label = nlp_output[0]["label"]

    return label
