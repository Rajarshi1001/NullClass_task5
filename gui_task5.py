# import libraries
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from keras.layers import TextVectorization
import re
import requests
# import tensorflow.strings as tf_strings
import json
import string
from keras.models import load_model
from tensorflow import argmax
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import tensorflow as tf

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

## loading the tokenizers
model_hi = load_model('english_to_hindi_lstm_model')
# model_fr = load_model('english_to_french_lstm_model')

#load Tokenizers

with open('english_tokenizer_hindi.json') as f:
    data = json.load(f)
    english_tokenizer_hindi = tokenizer_from_json(data)

with open('hindi_tokenizer.json') as f:
    data = json.load(f)
    hindi_tokenizer = tokenizer_from_json(data)


max_decoded_sentence_length = 20
    
    
with open('sequence_length_hindi.json') as f:
    max_length_hindi = json.load(f)
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_hindi(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer_hindi.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length_hindi)
    
    english_sentence = english_sentence.reshape((-1,max_length_hindi))
    
    hindi_sentence = model_hi.predict(english_sentence)[0]
    
    hindi_sentence = [np.argmax(word) for word in hindi_sentence]

    hindi_sentence = hindi_tokenizer.sequences_to_texts([hindi_sentence])[0]
    
    # print("hindi translation: ", hindi_sentence)
    
    return hindi_sentence

# Function to check if a word starts with a vowel
def starts_with_vowel(word):
    vowels = 'AEIOUaeiou'
    return word[0] in vowels

def translate(english_text):
    
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        'client': 'gtx',
        'sl': 'en',  
        'tl': 'hi',  
        'dt': 't',
        'q': english_text  
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        translation = response.json()[0][0][0]
        translated_sent = f"Hindi: {translation}"
    except Exception as e:
        translated_sent = "Error"
        
    return translated_sent

# Function to translate English word to Hindi
def solve():
    
    current_time = datetime.now()
    current_hour = current_time.hour
    word = input_entry.get()
    
    if starts_with_vowel(word):
        if current_hour < 21 or current_hour > 21:
            translation = "This word starts with Vowels, provide some other words and this model should be able to convert english word starts with vowels around 9 PM to 10 PM" 
        else:
            translation = translate(word)
    else:
        translation = translate(word)

    result_label.config(text = translation)

root = tk.Tk()
root.title("English Audio Translator")
root.geometry("500x300")

font = ('Helvetica', 14)

input_entry = tk.Entry(root, width=80, font=font)
input_entry.pack(pady=10)

instruction_label = tk.Label(root, text="Enter word for English to Hindi translation", wraplength=400, font=font)
instruction_label.pack(pady=10)

translate_button = tk.Button(root, text="Translate", command=solve, font=font)
translate_button.pack(pady=10)

result_label = tk.Label(root, text="", wraplength=400, font=font)
result_label.pack(pady=20)

root.mainloop()