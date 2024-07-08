# import libraries
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import pytz
import speech_recognition as sr
from keras.layers import TextVectorization
import re
# import tensorflow.strings as tf_strings
import json
import string
from keras.models import load_model
from tensorflow import argmax
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import tensorflow as tf



## loading the model

model_hi = load_model('english_to_hindi_lstm_model')
model_fr = load_model('english_to_french_lstm_model')

#load Tokenizers

with open('english_tokenizer_hindi.json') as f:
    data = json.load(f)
    english_tokenizer_hindi = tokenizer_from_json(data)

with open('hindi_tokenizer.json') as f:
    data = json.load(f)
    hindi_tokenizer = tokenizer_from_json(data)
    
with open('english_tokenizer.json') as f:
    data = json.load(f)
    english_tokenizer = tokenizer_from_json(data)

with open('french_tokenizer.json') as f:
    data = json.load(f)
    french_tokenizer = tokenizer_from_json(data)


max_decoded_sentence_length = 20

with open('sequence_length_hindi.json') as f:
    max_length_hindi = json.load(f)
    
with open('sequence_length.json') as f:
    max_length = json.load(f)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def beam_search_decoder(predictions, beam_width=3, epsilon=1e-10):
    sequences = [[[], 0.0]]  
    # Walk over each step in the sequence
    for row in predictions:
        all_candidates = list()
        # Expand each current candidate
        for seq, score in sequences:
            for j, prob in enumerate(row):
                prob = max(prob, epsilon)  # Ensure prob is non-zero
                candidate = [seq + [j], score - np.log(prob)]
                all_candidates.append(candidate)
        # Order all candidates by score (lowest score first)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # Select k best samples
        sequences = ordered[:beam_width]
    return sequences

with open('sequence_length.json') as f:
    max_length = json.load(f)
    
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def translate_to_french(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    
    english_sentence = english_sentence.reshape((-1,max_length))
    
    french_sentence = model_fr.predict(english_sentence)[0]
    
    french_sentence = [np.argmax(word) for word in french_sentence]

    french_sentence = french_tokenizer.sequences_to_texts([french_sentence])[0]
    
    # print("French translation: ", french_sentence)
    
    return french_sentence

def translate_to_french_beam_search(english_sentence, beam_width=3):
    english_sentence = english_sentence.lower()
    
    # Remove punctuation
    for punct in ['.', '?', '!', ',']:
        english_sentence = english_sentence.replace(punct, '')

    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length)
    english_sentence = english_sentence.reshape((-1, max_length))   
    predictions = model_fr.predict(english_sentence)[0]
    
    beam_results = beam_search_decoder(predictions, beam_width)
    
    # selecting the best result from beam search outputs
    best_sequence = beam_results[0][0]
    
    french_sentence = french_tokenizer.sequences_to_texts([best_sequence])[0]
    
    # print("French translation: ", french_sentence)
    
    return french_sentence
    
def translate_to_hindi(english_sentence):
    english_sentence = english_sentence.lower()
    
    english_sentence = english_sentence.replace(".", '')
    english_sentence = english_sentence.replace("?", '')
    english_sentence = english_sentence.replace("!", '')
    english_sentence = english_sentence.replace(",", '')
    
    english_sentence = english_tokenizer.texts_to_sequences([english_sentence])
    english_sentence = pad(english_sentence, max_length_hindi)
    
    english_sentence = english_sentence.reshape((-1,max_length_hindi))
    
    hindi_sentence = model_hi.predict(english_sentence)[0]
    
    hindi_sentence = [np.argmax(word) for word in hindi_sentence]

    hindi_sentence = hindi_tokenizer.sequences_to_texts([hindi_sentence])[0]
    
    # print("hindi translation: ", hindi_sentence)
    
    return hindi_sentence

## task 4 functions

def is_after_6pm_ist():
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    return current_time.hour >= 18

def capture_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    return audio

def listen_and_translate(audio=None):
    if audio is None:
        return "No audio input provided."

    recognizer = sr.Recognizer()

    try:
        english_text = recognizer.recognize_google(audio)
        print(f"You said: {english_text}")

        if not is_after_6pm_ist():
            return "Please try after 6 PM IST."

        if english_text.strip().lower()[0] in ['m', 'o']:
            return "Cannot translate words starting with 'M' or 'O'."

        hindi_translation = translate_to_hindi(english_text)
        return hindi_translation

    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio. Please try again."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
            
        
## Task 5 functions

# Function to check if a word starts with a vowel
def starts_with_vowel(word):
    vowels = 'AEIOUaeiou'
    return word[0] in vowels

# Function to translate English word to Hindi
def translate(word):
    
    current_time = datetime.now()
    current_hour = current_time.hour
    
    if starts_with_vowel(word):
        if current_hour < 21 or current_hour > 21:
            return "This word starts with Vowels, provide some other words." 
        else:
            translation = translate_to_hindi(word)
            return translation
    else:
        translation = translate_to_hindi(word)
        return translation


# Function to handle translation request based on selected language
def execute_option():
    selected_option = selected_option_var.get()
    translated_sent = ""

    if selected_option == "Task 4: English Audio Translation":
        instruction_label.config(text="Listening... Please speak now.")
        root.update()  # Update the GUI to show the new instruction
        audio = capture_audio()
        instruction_label.config(text="Processing...")
        root.update() 
        result = listen_and_translate(audio)
        translated_sent = f"Hindi : {result}"
    else:
        input_text = input_entry.get()
        
        if selected_option == "Task 1: English to French using pretrained LSTM":
            result = translate_to_french(input_text)
            translated_sent = f"French : {result}"
                   
        elif selected_option == "Task 2: Beam search decoding":
            result = translate_to_french_beam_search(input_text)
            translated_sent = f"French : {result}"

        elif selected_option == "Task 3: Dual Translation":
            if len(input_text) == 10:
                result_hi = translate_to_hindi(input_text)
                result_fr = translate_to_french(input_text)
                translated_sent = f"Hindi: {result_hi}\nFrench: {result_fr}"
            else:
                translated_sent = "Please provide a 10-letter English word."
            
        elif selected_option == "Task 5: Translation with vowels restrictions":
            result = translate(input_text)
            translated_sent = f"Hindi : {result}"
        
        else:
            translated_sent = "Please select a valid option."

    result_label.config(text=translated_sent)


# Setting up the main window
root = tk.Tk()
root.title("Language Translator")
root.geometry("500x300")

selected_option_var = tk.StringVar()

# Create an options box (drop-down menu)
options = ["Task 1: English to French using pretrained LSTM", 
           "Task 2: Beam search decoding", 
           "Task 3: Dual Translation", 
           "Task 4: English Audio Translation", 
           "Task 5: Translation with vowels restrictions"
        ]

def on_option_select(event):
    selected_option = selected_option_var.get()
    if selected_option == "Task 4: English Audio Translation":
        instruction_label.config(text="Click the Translate button to speak")
        input_entry.config(state='disabled')  # Disable text input for audio option
    else:
        instruction_label.config(text="")
        input_entry.config(state='normal')  # Enable text input for other options

option_menu = ttk.Combobox(root, textvariable=selected_option_var, values=options, width = 80)
option_menu.set("Select an option")  
option_menu.pack(pady=20)
option_menu.bind("<<ComboboxSelected>>", on_option_select)

input_entry = tk.Entry(root, width=80)
input_entry.pack(pady=10)

instruction_label = tk.Label(root, text="", wraplength=400)
instruction_label.pack(pady=10)

translate_button = tk.Button(root, text="Translate", command=execute_option)
translate_button.pack(pady=10)

result_label = tk.Label(root, text="", wraplength=400)
result_label.pack(pady=20)

root.mainloop()
