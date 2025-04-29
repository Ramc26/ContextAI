import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Setup device: use MPS if available (for Mac M2) else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ----------------------------
# Load models and tokenizers
# ----------------------------
# Contextual explanation model: Llama-3.2-1B-Instruct
EXPL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
expl_tokenizer = AutoTokenizer.from_pretrained(EXPL_MODEL, trust_remote_code=True)
expl_model = AutoModelForCausalLM.from_pretrained(
    EXPL_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device)
expl_model.eval()

# Translation model: facebook/nllb-200-distilled-600M
TRANS_MODEL = "facebook/nllb-200-distilled-600M"
trans_tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL).to(device)
# Identify Telugu BOS token id for forced language token
telugu_bos_id = trans_tokenizer.convert_tokens_to_ids("tel_Telu")

# Initialize Flask app
app = Flask(__name__)

@app.route("/explain_translate", methods=["POST"])
def explain_translate():
    """
    Accept JSON with 'paragraph' and 'text'.
    Returns a very brief English explanation (1-2 sentences) and its Telugu translation.
    """
    data = request.get_json() or {}
    paragraph = data.get("paragraph", "").strip()
    text = data.get("text", "").strip()
    if not paragraph or not text:
        return jsonify({"error": "Both 'paragraph' and 'text' are required."}), 400

    # Build prompt for a concise explanation
    prompt = (
        "You are a helpful assistant. Given the context below, briefly explain the following sentence/word in one or two sentences:\n\n"
        f"Context: {paragraph}\n"
        f"Sentence: \"{text}\"\n\n"
        "Brief Explanation:")

    # Generate concise explanation
    inputs = expl_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = expl_model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.3,
        eos_token_id=expl_tokenizer.eos_token_id,
        do_sample=False
    )
    decoded = expl_tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = decoded[len(prompt):].strip()

    # Translate to Telugu
    trans_inputs = trans_tokenizer(explanation, return_tensors="pt").to(device)
    trans_outputs = trans_model.generate(
        **trans_inputs,
        forced_bos_token_id=telugu_bos_id,
        max_length=128
    )
    translation = trans_tokenizer.decode(trans_outputs[0], skip_special_tokens=True)

    return jsonify({
        "explanation_en": explanation,
        "translation_te": translation
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6000)), debug=True)
