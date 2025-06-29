# File: digitalization_opportunity_extractor.py

import fitz  # PyMuPDF
import pytesseract
import io
from PIL import Image
import pandas as pd
import asyncio
import os
import nltk
import json
import spacy
import cv2
import numpy as np
import re
from tabulate import tabulate
from tqdm import tqdm
from fpdf import FPDF
from openai import OpenAI
from difflib import SequenceMatcher
from langdetect import detect
import time
import unicodedata
import math
from bs4 import BeautifulSoup
import requests
import nest_asyncio
nest_asyncio.apply()
import tiktoken
from openpyxl.utils import get_column_letter



# Load spaCy model for better sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Windows-specific Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Create OpenAI client
client = OpenAI(api_key="")




# GPT-4o-mini Request with retry logic and dynamic wait handling
async def ask_gpt4o(batch_sentences, page_number, semaphore, client):
    joined_text = "\n".join(batch_sentences)
    prompt = f"""
You are a top-level digital transformation consultant, specializing in the analysis of corporate sustainability reports. 
Your job is to accurately identify specific areas where manual processes, inefficiencies, or outdated practices are described 
and where digitalization could realistically improve operations.

Context:
- The sustainability reports often describe processes like manual data entry, physical attendance recording, email-based communication, spreadsheet-based reporting, etc.
- Your task is to carefully scan the provided statements to find REAL digitalization opportunities.

Rules:
- If a process is already digital, ignore it.
- If a sentence talks about something operational but has no inefficiency or manual work mentioned, ignore it.
- Only flag issues where:
  * Manual work is being done (e.g., manually recording data, training done physically, forms sent via email, using logbooks, etc.)
  * Inefficiencies are obvious (e.g., delays, lack of real-time monitoring, redundant paperwork)

Examples:
- "Training is delivered in person and attendance is marked manually." 
  → Area: HR Training 
  → Digital Solution: Online Training Portal with Digital Attendance
  → Reason: Automates training and attendance recording, improving efficiency.

- "Emissions are calculated manually using spreadsheets."
  → Area: Sustainability Reporting
  → Digital Solution: Automated Emission Calculator System
  → Reason: Reduces human error and speeds up reporting.

Instructions:
- Analyze each provided sentence or paragraph.
- Current Page: {page_number}
- Output ONLY when a real digitalization need is present.
- For each finding, return a JSON object containing:
  - 'search_key' : Atmost 3 search keys that were used to search for the finding (Make sure the search keys are derived before the evidence defined underneath)
  - 'page': The page number where it was found
  - 'evidence': The exact sentence or phrase from the input text
  - 'area': Related operational/business area
  - 'digital_solution': Specific digital solution to solve it
  - 'reason': Short 1-sentence justification why the solution fits
  - 'confidence': Rate your confidence in the need for digitalization on a scale of 1 (low) to 10 (very high)

- If no need is detected in the given text, return an empty list [].

Output:
Only output the JSON list format.
Be strict, precise, and professional.
If unsure, better to SKIP than to hallucinate a solution.

Statements:
{joined_text}
"""
    retries = 5

    async with semaphore:
        for attempt in range(retries):
            try:
                response = await asyncio.to_thread(tracked_chat_completion,
                    client=client,
                    model="gpt-4o-mini",  # <-- Changed from gpt-4o to gpt-4o-mini
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1000
                )
                content = response.choices[0].message.content.strip()

                if content.startswith("```"):
                    content = content.split("```", 2)[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()

                return json.loads(content)

            except Exception as e:
                print(f"Error in GPT-4o-mini Response on page {page_number}: {e}")
                error_message = str(e)
                match = re.search(r'try again in (\d+(\.\d+)?)s', error_message)
                if match:
                    wait_time = float(match.group(1)) + 0.5
                else:
                    wait_time = 5

                if attempt < retries - 1:
                    print(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    return []








# GPT-4o-mini Translation
async def translate_text_with_gpt(text, source_lang):
    prompt = f"""
You are a professional translator.
Translate the following text from {source_lang} to English precisely, without changing the meaning or omitting any information.
Keep the text structure clear and readable.

Text:
{text}
"""
    try:
        response = await asyncio.to_thread(tracked_chat_completion,
            client=client,
            model="gpt-4o-mini",  # <-- Changed from gpt-4o to gpt-4o-mini
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"Error during translation: {e}")
        return text




# OCR Preprocessing
def preprocess_image_for_ocr(image_bytes):
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert('L'))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return pytesseract.image_to_string(img)




# Extract text from pages in memory
def extract_text_from_pages(doc, start_page, end_page):
    pages = []
    for page_num in range(start_page, end_page):
        page = doc[page_num]
        text = page.get_text("text")
        if not text.strip():
            pix = page.get_pixmap()
            text = preprocess_image_for_ocr(pix.tobytes())
        pages.append((page_num + 1, text))
    return pages



# Smart Split using spaCy
def smart_sentence_split(text):
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        clean_sent = sent.text.strip()
        if len(clean_sent) > 10:
            sentences.append(clean_sent)
    return sentences







# Deduplicate similar suggestions (based on evidence text)
def deduplicate_findings(findings, threshold=0.8):
    unique = []
    for item in findings:
        duplicate = False
        for u_item in unique:
            if SequenceMatcher(None, item['evidence'], u_item['evidence']).ratio() > threshold:
                duplicate = True
                break
        if not duplicate:
            unique.append(item)
    return unique







# Step 1: Scrape full service page content (in plain text)
def fetch_ergosign_services_text():
    url = "https://www.ergosign.de/en/"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[:8000]  # limit context for GPT
    except Exception as e:
        print(f"Failed to fetch site: {e}")
        return ""








# Step 2: Map one row dynamically by analyzing live site + input
async def map_opportunity_to_service_dynamic(row, client):
    site_text = fetch_ergosign_services_text()
    prompt = f"""
You are a digital strategy expert at Ergosign.

Your task is to match a company's digitalization need with the most suitable real-world Ergosign service.
You are given the company's specific need and a full snapshot of Ergosign's website.

You MUST ONLY use the information from the site content. Do NOT invent services.
Give the exact service name that fits, and a brief justification (1 sentence) of how Ergosign can help with that service.

⚠ Important:
- Do NOT restate the suggested digitalization or service name in the justification.

Company's Need:
Suggested Digitalization: {row['Suggested Digitalization']}
Reason: {row['Reason']}

Ergosign Website Content:
{site_text}

Return a JSON like:
{{
  "matched_service": "<real service name from the site>",
  "mapping_justification": "<how and why this fits and how Ergosign helps>"
}}
"""

    try:
        response = await asyncio.to_thread(tracked_chat_completion,
            client=client,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
        parsed = json.loads(reply.split("```json")[-1].replace("```", "").strip())
        return parsed.get("matched_service", "N/A"), parsed.get("mapping_justification", "N/A")
    except Exception as e:
        print(f"Mapping failed: {e}")
        return "N/A", "Error or no match"








# Step 3: Map all rows
async def map_all_opportunities(df):
    tasks = [map_opportunity_to_service_dynamic(row, client) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    df["Mapped Ergosign Service"] = [r[0] for r in results]
    df["Mapping Justification"] = [r[1] for r in results]

    df = df[[
        "Suggested Digitalization",
        "Mapped Ergosign Service",
        "Mapping Justification"
    ]]

    # Write to Excel with predefined column widths
    output_file = "mapped_services.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Service Mapping")
        worksheet = writer.sheets["Service Mapping"]

        # Define column widths (adjust as needed)
        column_widths = [50, 40, 200]  # width in characters
        for i, width in enumerate(column_widths, start=1):
            col_letter = get_column_letter(i)
            worksheet.column_dimensions[col_letter].width = width
    print("\n🎯 Final Ergosign Service Mapping Table:\n")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))

    return df






# Translate only required fields for the Opportunities Table
async def translate_digitalization_findings(rows, source_lang="auto"):
    translated = []
    for row in rows:
        for col in ['search_key', 'evidence', 'area', 'digital_solution', 'reason']:
            row[col] = await translate_text_with_gpt(row[col], source_lang)
        translated.append(row)
    return translated









# Pricing for GPT-4o-mini
MODEL = "gpt-4o"
COST_INPUT_PER_1K = 0.00015
COST_OUTPUT_PER_1K = 0.0006

# Global counters
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0



# Token estimation
def count_tokens(messages, model=MODEL):
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0
    for message in messages:
        tokens += len(encoding.encode(message.get("content", "")))
    return tokens





# Wrap GPT call with cost tracking
def tracked_chat_completion(messages, temperature=0.3, model=MODEL, client=None, **kwargs):
    global total_input_tokens, total_output_tokens, total_cost

    if client is None:
        client = OpenAI()  # Uses OPENAI_API_KEY from environment

    input_tokens = count_tokens(messages, model)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs
    )

    output_content = response.choices[0].message.content
    output_tokens = len(tiktoken.encoding_for_model(model).encode(output_content))

    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

    input_cost = (input_tokens / 1000) * COST_INPUT_PER_1K
    output_cost = (output_tokens / 1000) * COST_OUTPUT_PER_1K
    total_cost += input_cost + output_cost

    return response





def print_cost_summary():
    print("\n----- GPT-4o Token Usage Summary -----")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Total Estimated Cost: ${total_cost:.4f}")
    print("-------------------------------------")





