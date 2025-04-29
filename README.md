# 🤖💡 ContextAI 🤖💡

**Author:**  
__Name__: Ram Bikkina  
__Email__: rakeshchowdary885@gmail.com

---

## Project Overview

**ContextAI** is a Flask-based microservice that takes:

1. A **paragraph** (context)  
2. A **sentence** (target phrase)

and returns:

- A **contextual English explanation** of the sentence, based on the paragraph  
- A **natural, conversational Telugu translation** of that explanation  

Under the hood, it uses Google’s Gemini-2.0-Flash model (via `langchain-google-genai`) for both steps. This prototype runs on macOS (M1/M2) with MPS support or CPU fallback.

---

## Features

- **`/explain_translate`** endpoint (POST)  
  - Input: JSON `{ "paragraph": "...", "sentence": "..." }`  
  - Output: JSON `{ "explanation_en": "...", "translation_te": "..." }`
- **`/health`** endpoint (GET) for quick uptime checks  
- Structured logging, error handling, and environment configuration  

---

## Prerequisites

- **macOS** (Apple Silicon recommended) or any OS with Python 3.12+  
- Git  
- Internet access for model downloads  
- Google Cloud API Key with access to Gemini models  

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/ramc26/ContextAI.git
   cd ContextAI
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**  
   Create a file named `.env` in the project root:
   ```bash
   touch .env
   ```
   Add your Google API key:
   ```env
   GOOGLE_API_KEY= XXXXXXXXXXXXXXXX
   ```
   > **Note:** Never commit your real key to Git. Use a placeholder or Git ignore `.env`.

---

## Running the Service

1. **Start the Flask server**  
   ```bash
   python main.py
   ```
   By default it will run on `http://0.0.0.0:5000/`.

2. **Health check**  
   ```bash
   curl http://localhost:5000/health
   # Response: {"status":"ok"}
   ```

3. **Explain & Translate**  
   ```bash
   curl -X POST http://localhost:5000/explain_translate \
     -H "Content-Type: application/json" \
     -d '{
           "paragraph": "The weather in Hyderabad today is mostly sunny with a current temperature of 36°C. The forecast for the day indicates that it will remain sunny, with the high also reaching 36°C and the low settling around 26°C. There is no rain expected today, and the wind is a gentle 6 km/h blowing from the southeast. Currently, it feels like 37°C due to the level of humidity in the air.",
           "sentence":  "feels like"
         }'
   ```
   **Expected response** (excerpt):
   ```json
   {
     "explanation_en": "The sentence \"feels like\" in the context of the paragraph is describing the **perceived temperature** or **apparent temperature**. While the actual measured temperature is 36°C, the level of humidity makes it *feel* hotter, specifically 37°C. This is because high humidity reduces the body's ability to cool itself through sweating, making the air feel warmer than it actually is.",
     "translation_te": "వాతావరణం 36°C ఉన్నా, చెమట వల్ల 37°C లా అనిపిస్తుంది."
   }
   ```

---

## Sample Test Cases

1. **Basic context + sentence**  
   ```bash
   curl -X POST http://localhost:5000/explain_translate \
     -H "Content-Type: application/json" \
     -d '{
           "paragraph": "Renewable energy sources like solar and wind power are gaining prominence as the world seeks to reduce its reliance on fossil fuels. These sources harness naturally replenishing energy, offering a cleaner alternative for electricity generation. While the initial investment in infrastructure can be substantial, the long-term operational costs are often lower, and they contribute significantly to mitigating climate change. The intermittency of these sources, however, presents a challenge for grid stability.",
           "sentence":  "The intermittency of these sources, however, presents a challenge for grid stability"
         }'
   ```
2. **Rule-based vs. deep learning context**  
   ```bash
   curl -X POST http://localhost:5000/explain_translate \
     -H "Content-Type: application/json" \
     -d '{
           "paragraph": "Simple rule-based systems follow if-then rules, while deep learning models learn patterns from data.",
           "sentence":  "The rule-based agent performed predictably under all test cases."
         }'
   ```
3. **Error handling (missing fields)**  
   ```bash
   curl -X POST http://localhost:5000/explain_translate \
     -H "Content-Type: application/json" \
     -d '{"paragraph": "Only one field"}'
   # Should return HTTP 400 with {"error":"Invalid JSON payload — 'paragraph' and 'sentence' required"}
   ```

---

## Notes & Next Steps

- For production deployments, swap out Flask’s built-in server for Gunicorn/Uvicorn.  
- Consider caching explanations for repeated queries.  
- Extend support to other languages by swapping the second-model prompt.  

Enjoy prototyping! 🚀