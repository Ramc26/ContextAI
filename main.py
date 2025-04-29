import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY. Please set it in your environment.")
    raise RuntimeError("GOOGLE_API_KEY not set")

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_retries=2
    )
    logger.info("ChatGoogleGenerativeAI client initialized successfully.")
except Exception:
    logger.exception("Failed to initialize ChatGoogleGenerativeAI client.")
    raise

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify(status="ok"), 200

@app.route("/explain_translate", methods=["POST"])
def explain_translate():
    """
    Expects JSON payload:
    {
        "paragraph": "<context paragraph>",
        "sentence":  "<sentence to explain and translate>"
    }
    Returns:
    {
        "explanation_en": "...",
        "translation_te": "..."
    }
    """
    try:
        data = request.get_json(force=True)
        paragraph = data["paragraph"]
        sentence  = data["sentence"]
    except (TypeError, KeyError):
        logger.warning("Invalid request payload: %s", request.data)
        return jsonify(error="Invalid JSON payload — 'paragraph' and 'sentence' required"), 400

    try:
        # 1. Contextual explanation in English
        logger.info("Generating contextual explanation for sentence.")
        con_expl = llm.invoke([
            ("system", "You are a helpful assistant that explains sentences in context."),
            ("human", (
                f"Paragraph: {paragraph}\n"
                f"Sentence: {sentence}\n"
                "Task: Explain the sentence in context of the paragraph."
            ))
        ]).content

        # 2. Translation to Telugu
        logger.info("Translating explanation to Telugu.")
        tel = llm.invoke([
            ("system", (
                "You are a helpful assistant that translates English into natural simple daily-use conversational Telugu. "
                "Output only the translated Telugu sentence with no additional text."
            )),
            ("human", con_expl)
        ]).content

        response = {
            "explanation_en": con_expl.strip(),
            "translation_te": tel.strip()
        }
        logger.info("Successfully generated explanation and translation.")
        return jsonify(response), 200

    except Exception:
        logger.exception("Error during explain_translate processing.")
        return jsonify(error="Internal server error"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=False)
