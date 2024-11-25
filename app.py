from flask import Flask, render_template, request
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('C:/Users/haziq.mirza/Desktop/medical CHATBOT/intents.json', 'r') as json_data:
    intents = json.load(json_data)


MODEL_FILE = "model.pth"
data = torch.load(MODEL_FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "E.D.I.T.H"

def response(sentence):

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    confidence = round(prob.item() * 100, 1)

    
    if prob.item() > 0.70:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return f"{bot_name} {confidence}%: {random.choice(intent['responses'])}"
    else:
        return (f"{bot_name} {confidence}%: I am confused. Please rephrase your question. "
                "Make sure it is related to the following topics: "
                "1) Guide about Vaccination Dose List, "
                "2) Side Effects of Vaccine, "
                "3) Hospitals for Vaccination, "
                "4) Information about vaccines.")


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    sentence = request.form["msg"]
    if sentence.lower() == "quit":
        return f"{bot_name}: Bye"
    else:
        return response(sentence)

if __name__ == "__main__":
    app.debug = True
    app.run()
