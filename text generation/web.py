from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

tokenizer = pickle.load(open('tokeniser.pkl', 'rb'))
model = load_model('text_gen_model.h5')

def generate_text(model, tokenizer, start_sequence, num_words=50):
    for _ in range(num_words):
        # Tokenize the input sequence
        token_list = tokenizer.texts_to_sequences([start_sequence])[0]
        token_list = tf.expand_dims(token_list, 0)  # Add batch dimension
        
        # Predict next token
        predicted = model.predict(token_list, verbose=0)
        next_token = tf.argmax(predicted[0]).numpy()
        
        # Convert token to word and append to sequence
        word = tokenizer.index_word.get(next_token, '')  # Use get to handle unknown tokens
        if word:
            start_sequence += ' ' + word  # Add the word to the sequence
        else:
            break
    
    return start_sequence

@app.route('/', methods=['GET', 'POST'])
def generate():
    paragraph = None  # Initialize paragraph variable
    if request.method == 'POST':
        input_text = request.form.get('text')
        if not input_text:
            return jsonify({'error': 'Input text is required'}), 400

        # Call the generate_text function with the input_text
        paragraph = generate_text(model, tokenizer, input_text, 50)
    
    # Render the home page and pass the generated paragraph (if any)
    return render_template('home.html', paragraph=paragraph)

if __name__ == '__main__':
    app.run(debug=True)
