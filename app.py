# app.py (updated)
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch

app = Flask(__name__)

def load_models():
    try:
        # Contextual Explanation Model
        llama_name ="meta-llama/Llama-3.2-1B-Instruct"

        llama_tokenizer = AutoTokenizer.from_pretrained(llama_name)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token  # Set pad token
        
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Translation Model
        nllb_name = "facebook/nllb-200-distilled-600M"
        nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_name)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        return llama_tokenizer, llama_model, nllb_tokenizer, nllb_model

    except Exception as e:
        print(f"Error loading models: {e}")
        raise

llama_tokenizer, llama_model, nllb_tokenizer, nllb_model = load_models()

def generate_contextual_explanation(context, sentence):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Provide a contextual explanation of the given sentence based on the provided paragraph."},
            {"role": "user", "content": f"Context:\n{context}\n\nSentence to explain:\n{sentence}\n\nContextual Explanation:"}
        ]

        inputs = llama_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(llama_model.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs).to(llama_model.device)

        outputs = llama_model.generate(
            inputs,
            attention_mask=attention_mask,
            pad_token_id=llama_tokenizer.pad_token_id,  # Explicit pad token
            eos_token_id=llama_tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        return llama_tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Explanation generation error: {e}")
        return ""

def translate_to_telugu(text):
    try:
        translation_pipeline = pipeline(
            "translation",
            model=nllb_model,
            tokenizer=nllb_tokenizer,
            src_lang="eng_Latn",
            tgt_lang="tel_Telu",
            max_length=400,
            device_map="auto"
        )
        result = translation_pipeline(text)
        return result[0]['translation_text']
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

@app.route('/explain_translate', methods=['POST'])
def handle_translation():
    data = request.get_json()
    
    if not data or 'paragraph' not in data or 'sentence' not in data:
        return jsonify({"error": "Invalid request format"}), 400
    
    context = data['paragraph']
    sentence = data['sentence']
    print(f"Context: {context}\n Sentence: {sentence}")
    ai_desc = generate_contextual_explanation(context, sentence)

    print(f"AI Description: {ai_desc}")
    if not ai_desc:
        return jsonify({"error": "Failed to generate explanation"}), 500
    
    telugu_translation = translate_to_telugu(ai_desc)

    print(f"Telugu Translation: {telugu_translation}")

    if not telugu_translation:
        return jsonify({"error": "Failed to translate text"}), 500
    
    return jsonify({
        "contextual_explanation": ai_desc,
        "telugu_translation": telugu_translation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)