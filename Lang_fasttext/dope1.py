import os
import fasttext
import langid

# Load the pre-trained FastText model
model_path = 'D:/fastx/lid.176.bin'
if not os.path.isfile(model_path):
    raise ValueError(f"Model file not found at {model_path}")

model = fasttext.load_model(model_path)

def detect_language_with_fasttext(word):
    predictions = model.predict(word, k=1)  # Top-1 prediction
    return predictions[0][0].replace('_label_', '')

def detect_language_with_langid(word):
    lang, _ = langid.classify(word)
    return lang

def detect_language_of_words(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        text = infile.read()
    
    words = text.split()
    detected_words = []

    for word in words:
        fasttext_lang = detect_language_with_fasttext(word)
        langid_lang = detect_language_with_langid(word)
        
        # Majority vote or confidence score approach
        if fasttext_lang == langid_lang:
            detected_lang = fasttext_lang
        else:
            # Apply more sophisticated logic if needed
            detected_lang = fasttext_lang  # For simplicity, using FastText result here

        detected_words.append(f"{word} ({detected_lang})")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(' '.join(detected_words))

# Replace 'input.txt' and 'output.txt' with your file paths
#"Give your input file in single quotes (eg input.txt)","give your output file in single quote (eg output.txt)"
detect_language_of_words ('','')
