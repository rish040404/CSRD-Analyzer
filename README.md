
# 🌍 CSRD Sustainability Report Analyzer

This project helps identify areas for digital improvements in sustainability reports, especially those aligned with the Corporate Sustainability Reporting Directive (CSRD). By combining Optical Character Recognition (OCR), Natural Language Processing (NLP), and Generative AI, it scans lengthy documents and extracts practical digitalization suggestions. It then maps these opportunities to real services offered by Ergosign.

---

## ✨ What It Does

- Upload a sustainability report in PDF format
- Automatically read and extract content using OCR if needed
- Detect real opportunities where digital tools could replace outdated, manual practices
- Translate findings to English (if the report is in another language)
- Match insights to actual Ergosign services based on their public website
- Export the results in Excel format

---

## 🧩 Project Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | User interface to upload reports and run analysis |
| `pipeline.py` | Core logic that analyzes the report and detects needs |
| `service_mapper.py` | Maps each finding to a relevant Ergosign service |
| `utils.py` | Helper functions for OCR, translation, GPT calls, and mapping |

---

## 🚀 Getting Started

1. **Clone this repository** or download the ZIP:
   ```bash
   git clone https://github.com/your-username/csrd-analyzer.git
   cd csrd-analyzer
   ```

2. **(Optional)** Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (for scanned PDFs):
   - Download: https://github.com/tesseract-ocr/tesseract
   - Ensure the path is correct in `utils.py`:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```

5. **Enter in your API Key**:
   - Type your API Key into line 38 of the pipeline.py file and line 41 of the utils.py file.

7. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📦 Outputs

- `opportunities.xlsx` – a summary of all detected digital opportunities
- `mapped_services.xlsx` – suggestions mapped to Ergosign offerings

---

## 💡 Notes

- The current model used is `gpt-4o-mini`, chosen for efficiency and affordability
- The tool assumes access to the OpenAI API – remember to manage your API key securely
- OCR quality and language translation can affect the outcome, especially for scanned or non-English reports

---

## 🛠 Use Cases

- Consultants preparing digital strategy reports
- Analysts assessing compliance with CSRD
- Teams looking to automate insights from sustainability documentation

---

## 📜 License

This tool is intended for Ergosign employees' purposes only. Please review and comply with licensing for OpenAI, spaCy, Tesseract, and other dependencies.

---

## 🧭 How to Use the Tool (Step-by-Step)

Once the Streamlit app is running:

1. **Upload Report**  
   Drag and drop the CSRD sustainability report (PDF) of your choice into the upload box.

2. **Run the Pipeline**  
   Click on **"▶️ Run Pipeline (Digitalization Detection)"**.  
   This scans the report and identifies areas where digital solutions may be helpful.

3. **Download Opportunities**  
   Once the analysis finishes, click **"📅 Download Excel Report"** to save the `opportunities.xlsx` file.  
   This file contains the detected digitalization needs from your report.

4. **Map Services**  
   Next, click **"💼 Map Services to Digitalization Suggestions"**.  
   This connects your findings with real services from Ergosign.

5. **Download Mapped Services**  
   After mapping is complete, click **"📅 Download Mapped Services Excel"** to save the `mapped_services.xlsx` file.  
   This file contains the recommendations matched to Ergosign's service offerings.

6. **Track GPT Usage & Cost**  
   Click **"GPT cost tracker"** at any point to see a summary of tokens used and the estimated cost for GPT model usage.

These steps guide you through a full analysis cycle using the uploaded sustainability report.

---
