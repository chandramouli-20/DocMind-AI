import streamlit as st
import fitz
import os
import io
import re
import time
import numpy as np
import pandas as pd
import pytesseract

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            #18234a 0%,
            #0b1020 38%,
            #070b16 100%
        );
    color: #f5f7ff;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #ffffff;
}


/* HERO */

.hero {
    padding: 35px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(83, 102, 255, 0.30),
            rgba(18, 25, 55, 0.92)
        );
    border: 1px solid rgba(130, 145, 255, 0.28);
    margin-bottom: 30px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 19px;
    color: #c7d0f5;
    line-height: 1.6;
}


/* PIPELINE */

.pipeline {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 25px;
}

.pipeline-item {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 14px;
    padding: 11px 16px;
    font-size: 14px;
    color: #eef1ff;
}

.arrow {
    color: #8290ff;
    font-size: 22px;
    font-weight: bold;
}


/* STAT CARDS */

.stat-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    min-height: 115px;
}

.stat-number {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
}

.stat-label {
    font-size: 13px;
    color: #aeb8dd;
    margin-top: 5px;
}


/* ANSWER */

.answer-card {
    background:
        linear-gradient(
            135deg,
            rgba(34, 44, 92, 0.90),
            rgba(15, 21, 45, 0.95)
        );
    border: 1px solid rgba(116, 134, 255, 0.30);
    border-radius: 20px;
    padding: 25px;
    margin-top: 15px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.20);
}


/* SECTION */

.section-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-top: 15px;
}


/* EVIDENCE */

.evidence-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 12px;
}

.source-tag {
    display: inline-block;
    background: rgba(102,118,255,0.18);
    border: 1px solid rgba(102,118,255,0.30);
    padding: 5px 10px;
    border-radius: 10px;
    color: #dbe0ff;
    font-size: 12px;
}

.small-muted {
    color: #9fa9cb;
    font-size: 13px;
}


/* STATUS */

.success-box {
    padding: 15px;
    border-radius: 14px;
    background: rgba(40,180,110,0.10);
    border: 1px solid rgba(40,180,110,0.25);
}

.warning-box {
    padding: 15px;
    border-radius: 14px;
    background: rgba(255,180,50,0.10);
    border: 1px solid rgba(255,180,50,0.25);
}


/* FILE UPLOADER */

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.035);
    border-radius: 15px;
    padding: 10px;
}

</style>
""",
unsafe_allow_html=True
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY or API_KEY == "YOUR_REAL_KEY":

    st.error(
        "Gemini API key was not found."
    )

    st.info(
        "Make sure your .env file contains:"
    )

    st.code(
        "GEMINI_API_KEY=YOUR_API_KEY"
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

try:

    client = genai.Client(
        api_key=API_KEY
    )

except Exception as e:

    st.error(
        f"Gemini client initialization failed: {e}"
    )

    st.stop()


# ============================================================
# GEMINI MODELS
# ============================================================

# Primary model that we want to use
PRIMARY_MODEL = "gemini-3.5-flash"

# Backup models
BACKUP_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.7-flash"
]


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_PATH
    )


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "pages" not in st.session_state:
    st.session_state.pages = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "document_stats" not in st.session_state:
    st.session_state.document_stats = ({}, {})


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# OCR
# ============================================================

def perform_ocr(image):

    try:

        text = pytesseract.image_to_string(
            image
        )

        return text.strip()

    except Exception:

        return ""


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_page(page):

    try:

        matrix = fitz.Matrix(
            1.25,
            1.25
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image_bytes = pix.tobytes(
            "png"
        )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image

    except Exception:

        return None


# ============================================================
# COMPRESS IMAGE FOR GEMINI
# ============================================================

def prepare_image_for_gemini(image):

    try:

        img = image.copy()

        max_width = 1400

        if img.width > max_width:

            ratio = (
                max_width
                / img.width
            )

            new_height = int(
                img.height * ratio
            )

            img = img.resize(
                (
                    max_width,
                    new_height
                )
            )

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="JPEG",
            quality=75,
            optimize=True
        )

        return buffer.getvalue()

    except Exception:

        return None


# ============================================================
# EXTRACT TABLES
# ============================================================

def extract_tables(page):

    tables = []

    try:

        finder = page.find_tables()

        for table in finder.tables:

            try:

                df = table.to_pandas()

                if (
                    df is not None
                    and not df.empty
                ):

                    tables.append(df)

            except Exception:

                continue

    except Exception:

        pass

    return tables


# ============================================================
# TABLE TO TEXT
# ============================================================

def tables_to_text(tables):

    if not tables:

        return ""

    output = []

    for index, df in enumerate(tables):

        output.append(
            f"TABLE {index + 1}"
        )

        try:

            output.append(
                df.to_string(
                    index=False
                )
            )

        except Exception:

            output.append(
                str(df)
            )

        output.append("")

    return "\n".join(output)



# ============================================================
# DOCUMENT METADATA / STATISTICS
# ============================================================

def count_words(text):
    """Count words from extracted/OCR text."""
    if not text:
        return 0
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def get_document_statistics(pages):
    """Calculate exact statistics from all processed PDF pages."""
    stats_by_doc = {}

    for page in pages:
        doc = page["document"]

        if doc not in stats_by_doc:
            stats_by_doc[doc] = {
                "pages": 0,
                "words": 0,
                "characters": 0,
                "ocr_pages": 0,
                "tables": 0,
                "images": 0,
                "text_pages": 0,
                "empty_pages": 0
            }

        s = stats_by_doc[doc]
        s["pages"] += 1

        text = page.get("text", "") or ""
        s["words"] += count_words(text)
        s["characters"] += len(text)
        s["tables"] += len(page.get("tables", []))
        s["images"] += page.get("image_count", 0)

        if page.get("ocr_used", False):
            s["ocr_pages"] += 1

        if text.strip():
            s["text_pages"] += 1
        else:
            s["empty_pages"] += 1

    total = {
        "pages": sum(x["pages"] for x in stats_by_doc.values()),
        "words": sum(x["words"] for x in stats_by_doc.values()),
        "characters": sum(x["characters"] for x in stats_by_doc.values()),
        "ocr_pages": sum(x["ocr_pages"] for x in stats_by_doc.values()),
        "tables": sum(x["tables"] for x in stats_by_doc.values()),
        "images": sum(x["images"] for x in stats_by_doc.values()),
        "documents": len(stats_by_doc)
    }

    return stats_by_doc, total


def is_metadata_question(question):
    """Detect questions that need exact whole-document statistics."""
    q = question.lower().strip()

    metadata_terms = [
        "how many pages", "number of pages", "total pages", "page count",
        "how many words", "number of words", "total words", "word count",
        "how many characters", "number of characters", "character count",
        "how many letters", "total characters",
        "how many images", "number of images", "image count",
        "how many tables", "number of tables", "table count",
        "how many ocr", "ocr pages", "scanned pages", "number of scanned",
        "how many documents", "number of documents", "document count",
        "average words per page", "average words/page",
        "average characters per page",
        "document statistics", "document stats", "pdf statistics",
        "pdf stats", "metadata"
    ]

    return any(term in q for term in metadata_terms)


def answer_metadata_question(question, pages):
    """
    Answer document-wide statistical questions directly from Python.
    Gemini is intentionally NOT used for exact counts.
    """
    q = question.lower().strip()
    stats_by_doc, total = get_document_statistics(pages)

    # Determine whether the user is asking about one named document.
    selected_doc = None
    for doc in stats_by_doc:
        if doc.lower() in q:
            selected_doc = doc
            break

    if selected_doc:
        s = stats_by_doc[selected_doc]
        scope = f"**{selected_doc}**"
    else:
        s = total
        scope = "the uploaded documents"

    # Pages
    if any(x in q for x in [
        "how many pages", "number of pages", "total pages", "page count"
    ]):
        if selected_doc:
            return (
                f"**{scope} contains {s['pages']} page(s).**\n\n"
                f"Source: `{selected_doc}` — all pages were counted directly from the PDF."
            )
        return (
            f"**The uploaded documents contain {s['pages']} page(s) in total.**\n\n"
            "This count is calculated directly from the PDF files."
        )

    # Words
    if any(x in q for x in [
        "how many words", "number of words", "total words", "word count"
    ]):
        if selected_doc:
            return (
                f"**{scope} contains {s['words']:,} words.**\n\n"
                f"Source: `{selected_doc}` — text from normal and OCR pages was counted."
            )
        return (
            f"**The uploaded documents contain {s['words']:,} words in total.**\n\n"
            "The count includes text extracted from normal PDF pages and OCR text from scanned pages."
        )

    # Characters
    if any(x in q for x in [
        "how many characters", "number of characters", "character count",
        "how many letters", "total characters"
    ]):
        return (
            f"**{scope} contains {s['characters']:,} characters.**\n\n"
            "Characters are counted from the extracted/OCR text."
        )

    # Images
    if any(x in q for x in [
        "how many images", "number of images", "image count"
    ]):
        return (
            f"**{scope} contains {s['images']:,} embedded image(s).**\n\n"
            "This counts embedded PDF images, not the rendered page images used for OCR."
        )

    # Tables
    if any(x in q for x in [
        "how many tables", "number of tables", "table count"
    ]):
        return f"**{scope} contains {s['tables']:,} detected table(s).**"

    # OCR/scanned pages
    if any(x in q for x in [
        "how many ocr", "ocr pages", "scanned pages", "number of scanned"
    ]):
        return (
            f"**{scope} contains {s['ocr_pages']:,} page(s) processed using OCR.**\n\n"
            "OCR was used when the page had very little extractable PDF text."
        )

    # Documents
    if any(x in q for x in [
        "how many documents", "number of documents", "document count"
    ]):
        return f"**There are {s['documents']:,} uploaded document(s).**"

    # Average words/page
    if any(x in q for x in [
        "average words per page", "average words/page"
    ]):
        avg = s["words"] / s["pages"] if s["pages"] else 0
        return f"**Average words per page: {avg:,.2f} words/page.**"

    # Average chars/page
    if "average characters per page" in q:
        avg = s["characters"] / s["pages"] if s["pages"] else 0
        return f"**Average characters per page: {avg:,.2f} characters/page.**"

    # General statistics
    if "stat" in q or "metadata" in q:
        document_count = len(stats_by_doc) if not selected_doc else 1
        return (
            f"### Document Statistics\n\n"
            f"- **Documents:** {document_count:,}\n"
            f"- **Pages:** {s['pages']:,}\n"
            f"- **Words:** {s['words']:,}\n"
            f"- **Characters:** {s['characters']:,}\n"
            f"- **OCR/Scanned Pages:** {s['ocr_pages']:,}\n"
            f"- **Detected Tables:** {s['tables']:,}\n"
            f"- **Embedded Images:** {s['images']:,}\n"
        )

    return None


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    document_name = uploaded_file.name

    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    document_pages = []

    for page_number in range(
        len(document)
    ):

        page = document[
            page_number
        ]

        # TEXT
        try:

            text = page.get_text(
                "text"
            ).strip()

        except Exception:

            text = ""

        # TABLES
        tables = extract_tables(
            page
        )

        table_text = tables_to_text(
            tables
        )

        # PAGE IMAGE
        image = render_page(
            page
        )

        # OCR
        ocr_used = False

        if (
            len(text) < 50
            and image is not None
        ):

            ocr_text = perform_ocr(
                image
            )

            if ocr_text:

                text = ocr_text

                ocr_used = True

        # Combined search content
        search_text = (
            text
            + "\n"
            + table_text
        )

        document_pages.append(
            {
                "document": document_name,
                "page": page_number + 1,
                "text": text,
                "tables": tables,
                "table_text": table_text,
                "search_text": search_text,
                "image": image,
                "ocr_used": ocr_used,
                "image_count": len(page.get_images(full=True))
            }
        )

    document.close()

    return document_pages


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(pages):

    texts = []

    for page in pages:

        text = page.get(
            "search_text",
            ""
        )

        if not text:

            text = (
                "Document: "
                + page["document"]
                + " Page: "
                + str(page["page"])
            )

        texts.append(
            text[:12000]
        )

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_pages(
    question,
    pages,
    embeddings,
    top_k=4
):

    if (
        not pages
        or embeddings is None
    ):

        return []

    question_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
    )

    scores = np.dot(
        embeddings,
        question_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        item = pages[index].copy()

        item["score"] = float(
            scores[index]
        )

        results.append(
            item
        )

    return results


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence_text(
    retrieved_pages
):

    evidence_sections = []

    for item in retrieved_pages:

        document = item[
            "document"
        ]

        page = item[
            "page"
        ]

        text = item.get(
            "text",
            ""
        )

        table_text = item.get(
            "table_text",
            ""
        )

        section = (
            f"DOCUMENT: {document}\n"
            f"PAGE: {page}\n"
        )

        if text:

            section += (
                "\nTEXT CONTENT:\n"
                + text[:10000]
            )

        if table_text:

            section += (
                "\n\nTABLE CONTENT:\n"
                + table_text[:10000]
            )

        evidence_sections.append(
            section
        )

    return (
        "\n\n"
        + (
            "\n\n"
            + "=" * 70
            + "\n\n"
        ).join(
            evidence_sections
        )
    )


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_prompt(
    question,
    evidence
):

    return f"""
You are DocMind AI.

You are an expert multimodal document analysis assistant.

Your job is to answer questions using the supplied evidence
from PDF documents.

USER QUESTION:

{question}


RETRIEVED EVIDENCE:

{evidence}


IMPORTANT INSTRUCTIONS:

1. Answer ONLY using information supported by the uploaded
documents.

2. Do not invent values, percentages, dates, names,
measurements, reasons, or conclusions.

3. Every important factual statement must include a citation.

Use exactly this citation format:

[Document: filename.pdf, Page: 2]

4. If a value comes from a table, cite the table's page.

5. If a value comes from a chart or graph, cite the page
containing the chart.

6. If the question asks for a comparison, show both values
before explaining the difference.

7. If a calculation is required, show the calculation.

8. If the question asks for reasons, identify the reasons
that are actually supported by the evidence.

9. Do not create reasons simply because they sound logical.

10. Use the page images to inspect charts, graphs, diagrams,
and visual information when available.

11. If the evidence does not contain enough information,
write:

Insufficient evidence in the retrieved documents.

12. Do not cite irrelevant pages.

13. Give a clear structured answer.

14. At the end provide:

CITATIONS:
- filename.pdf | Page 2
- filename.pdf | Page 5

Only list pages that directly support the answer.
"""


# ============================================================
# GEMINI API CALL
# ============================================================

def call_gemini(
    model_name,
    contents
):

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000
            )
        )

        if response is None:

            return None, "Empty response"

        if not response.text:

            return None, "Gemini returned no text"

        return response.text, None

    except Exception as e:

        return None, str(e)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_pages
):

    evidence = build_evidence_text(
        retrieved_pages
    )

    prompt = build_prompt(
        question,
        evidence
    )

    # Start with text
    contents = [
        types.Part.from_text(
            text=prompt
        )
    ]

    # --------------------------------------------------------
    # Add ONLY top 2 relevant page images
    # --------------------------------------------------------

    image_count = 0

    for item in retrieved_pages:

        if image_count >= 2:

            break

        image = item.get(
            "image"
        )

        if image is None:

            continue

        image_bytes = (
            prepare_image_for_gemini(
                image
            )
        )

        if not image_bytes:

            continue

        try:

            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                )
            )

            image_count += 1

        except Exception:

            continue

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = [
        PRIMARY_MODEL
    ] + BACKUP_MODELS

    errors = []

    # --------------------------------------------------------
    # Try models
    # --------------------------------------------------------

    for model_name in models:

        for attempt in range(2):

            try:

                response, error = call_gemini(
                    model_name,
                    contents
                )

                if response:

                    return response

                errors.append(
                    f"{model_name} "
                    f"(attempt {attempt + 1}): "
                    f"{error}"
                )

                time.sleep(
                    2 * (attempt + 1)
                )

            except Exception as e:

                errors.append(
                    f"{model_name} "
                    f"(attempt {attempt + 1}): "
                    f"{str(e)}"
                )

                time.sleep(
                    2 * (attempt + 1)
                )

    # --------------------------------------------------------
    # All models failed
    # --------------------------------------------------------

    st.error(
        "❌ Gemini could not generate the answer."
    )

    with st.expander(
        "🔧 Gemini Error Details"
    ):

        for error in errors:

            st.code(
                error
            )

    return (
        "Gemini generation failed. "
        "Please check the Gemini error details above."
    )


# ============================================================
# PARSE CITATIONS
# ============================================================

def parse_citations(answer):

    citations = []

    if not answer:

        return citations

    # Citation block
    pattern = (
        r"-\s*(.*?)\s*\|\s*Page\s+(\d+)"
    )

    matches = re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE
    )

    for document, page in matches:

        citation = {
            "document": document.strip(),
            "page": int(page)
        }

        if citation not in citations:

            citations.append(
                citation
            )

    # Inline citations
    inline_pattern = (
        r"\[Document:\s*(.*?),\s*Page:\s*(\d+)\]"
    )

    inline_matches = re.findall(
        inline_pattern,
        answer,
        flags=re.IGNORECASE
    )

    for document, page in inline_matches:

        citation = {
            "document": document.strip(),
            "page": int(page)
        }

        if citation not in citations:

            citations.append(
                citation
            )

    return citations


# ============================================================
# DISPLAY SUPPORTING PAGES
# ============================================================

def display_supporting_pages(
    answer,
    retrieved_pages
):

    citations = parse_citations(
        answer
    )

    cited_pages = []

    for citation in citations:

        for item in retrieved_pages:

            if (
                item["document"].strip().lower()
                ==
                citation["document"].strip().lower()
                and
                item["page"]
                ==
                citation["page"]
            ):

                if item not in cited_pages:

                    cited_pages.append(
                        item
                    )

    if not cited_pages:

        return

    st.markdown(
        "### 🖼️ Supporting Evidence"
    )

    columns = st.columns(2)

    for index, item in enumerate(
        cited_pages
    ):

        with columns[
            index % 2
        ]:

            st.markdown(
f"""
<div class="evidence-card">

<div class="source-tag">
{item["document"]}
</div>

<br><br>

<b>📄 Page {item["page"]}</b>

</div>
""",
                unsafe_allow_html=True
            )

            image = item.get(
                "image"
            )

            if image:

                st.image(
                    image,
                    use_container_width=True
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
"""
# 🧠 DocMind AI

### Multimodal Document Intelligence

Upload PDF documents containing:

📄 Text

📊 Tables

📈 Charts

🖼️ Images

📑 Scanned pages

DocMind retrieves relevant evidence
and generates answers with document
and page citations.
"""
    )

    st.divider()

    st.markdown(
        "### ⚙️ Processing Pipeline"
    )

    st.markdown(
"""
**1. PDF Upload**

↓

**2. Text Extraction**

↓

**3. Table Detection**

↓

**4. OCR**

↓

**5. Page Images**

↓

**6. Semantic Retrieval**

↓

**7. Gemini Vision Analysis**

↓

**8. Evidence-Based Answer**
"""
    )

    st.divider()

    st.markdown(
        "### 🤖 Gemini"
    )

    st.caption(
        f"Primary: {PRIMARY_MODEL}"
    )

    st.caption(
        "Backup: Gemini 3.6 Flash"
    )

    st.caption(
        "Backup: Gemini 3.7 Flash"
    )

    st.divider()

    st.markdown(
        "### 🔐 Security"
    )

    st.caption(
        "API key is loaded from .env"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
"""
<div class="hero">

<div class="hero-title">
🧠 DocMind AI
</div>

<div class="hero-subtitle">
Multimodal Document Intelligence &amp;
Evidence-Based Question Answering
</div>

<div class="pipeline">

<div class="pipeline-item">
📄 Documents
</div>

<div class="arrow">
→
</div>

<div class="pipeline-item">
🔎 Extract
</div>

<div class="arrow">
→
</div>

<div class="pipeline-item">
🧩 Retrieve
</div>

<div class="arrow">
→
</div>

<div class="pipeline-item">
👁️ Analyze
</div>

<div class="arrow">
→
</div>

<div class="pipeline-item">
📌 Cite
</div>

</div>

</div>
""",
unsafe_allow_html=True
)


# ============================================================
# DOCUMENT WORKSPACE
# ============================================================

st.markdown(
    "## 📚 Document Workspace"
)

st.write(
    "Upload one or more PDF documents."
)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS BUTTON
# ============================================================

if uploaded_files:

    st.success(
        f"✅ {len(uploaded_files)} PDF file(s) selected."
    )

    if st.button(
        "🚀 Process Documents",
        type="primary",
        use_container_width=True
    ):

        st.session_state.documents = []

        st.session_state.pages = []

        st.session_state.embeddings = None

        st.session_state.processed = False
        st.session_state.document_stats = ({}, {})

        all_pages = []

        progress = st.progress(
            0
        )

        status = st.empty()

        total = len(
            uploaded_files
        )

        for index, file in enumerate(
            uploaded_files
        ):

            status.write(
                f"📄 Processing {file.name}..."
            )

            try:

                pages = process_pdf(
                    file
                )

                all_pages.extend(
                    pages
                )

                st.session_state.documents.append(
                    file.name
                )

            except Exception as e:

                st.error(
                    f"Error processing "
                    f"{file.name}: {e}"
                )

            progress.progress(
                (index + 1) / total
            )

        if all_pages:

            status.write(
                "📊 Calculating document statistics..."
            )

            st.session_state.document_stats = (
                get_document_statistics(all_pages)
            )

            status.write(
                "🧩 Creating semantic embeddings..."
            )

            st.session_state.pages = (
                all_pages
            )

            st.session_state.embeddings = (
                create_embeddings(
                    all_pages
                )
            )

            st.session_state.processed = True

            status.success(
                "✅ Documents processed successfully."
            )

        else:

            st.error(
                "No PDF pages could be processed."
            )


# ============================================================
# WORKSPACE STATS
# ============================================================

if st.session_state.processed:

    pages = st.session_state.pages

    total_documents = len(
        st.session_state.documents
    )

    total_pages = len(
        pages
    )

    total_tables = sum(
        len(
            page.get(
                "tables",
                []
            )
        )
        for page in pages
    )

    total_ocr = sum(
        1
        for page in pages
        if page.get(
            "ocr_used",
            False
        )
    )

    st.markdown(
        "## 📊 Workspace Overview"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:

        st.markdown(
f"""
<div class="stat-card">
<div class="stat-number">
{total_documents}
</div>
<div class="stat-label">
Documents
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
f"""
<div class="stat-card">
<div class="stat-number">
{total_pages}
</div>
<div class="stat-label">
Pages
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
f"""
<div class="stat-card">
<div class="stat-number">
{total_tables}
</div>
<div class="stat-label">
Tables
</div>
</div>
""",
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
f"""
<div class="stat-card">
<div class="stat-number">
{total_ocr}
</div>
<div class="stat-label">
OCR Pages
</div>
</div>
""",
            unsafe_allow_html=True
        )

    total_words = sum(
        count_words(page.get("text", ""))
        for page in pages
    )

    with c5:
        st.markdown(
f"""
<div class="stat-card">
<div class="stat-number">
{total_words:,}
</div>
<div class="stat-label">
Words
</div>
</div>
""",
            unsafe_allow_html=True
        )

    total_images = sum(
        page.get("image_count", 0)
        for page in pages
    )

    with c6:
        st.markdown(
f"""
<div class="stat-card">
<div class="stat-number">
{total_images:,}
</div>
<div class="stat-label">
Images
</div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# LOADED DOCUMENTS
# ============================================================

if st.session_state.processed:

    st.markdown(
        "### 📄 Loaded Documents"
    )

    for document in (
        st.session_state.documents
    ):

        st.markdown(
f"""
<div class="section-card">
📘 <b>{document}</b>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# QUESTION SECTION
# ============================================================

if st.session_state.processed:

    st.markdown(
        "## 💬 Ask Your Documents"
    )

    st.write(
        "Ask questions about text, tables, charts, "
        "numbers, or multiple documents."
    )

    question = st.text_area(
        "Enter your question",
        height=140,
        placeholder=(
            "Example:\n"
            "Compare the production efficiency between Q2 and Q4. "
            "What are the three main reasons for the change? "
            "Provide the exact values and cite the document name "
            "and page number for each piece of evidence."
        )
    )

    if st.button(
        "🔍 Analyze Documents",
        type="primary",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ================================================
            # DOCUMENT-WIDE METADATA QUESTIONS
            # ================================================
            # Exact statistics are calculated locally from ALL
            # processed pages. Do not ask Gemini to estimate them.

            metadata_answer = None

            if is_metadata_question(question):
                metadata_answer = answer_metadata_question(
                    question,
                    st.session_state.pages
                )

            if metadata_answer:
                st.session_state.last_answer = metadata_answer

                st.markdown(
                    "## 🧠 Answer"
                )

                st.markdown(
                    f"""
<div class="answer-card">
""",
                    unsafe_allow_html=True
                )

                st.markdown(metadata_answer)

                st.markdown(
                    """
</div>
""",
                    unsafe_allow_html=True
                )

                st.info(
                    "📊 This answer was calculated directly from the uploaded PDF data, not generated by Gemini."
                )

            else:

                # ================================================
                # RETRIEVAL FOR CONTENT QUESTIONS
                # ================================================

                with st.spinner(
                    "🔎 Searching relevant pages..."
                ):

                    retrieved_pages = retrieve_pages(
                        question,
                        st.session_state.pages,
                        st.session_state.embeddings,
                        top_k=4
                    )

                if not retrieved_pages:

                    st.error(
                        "No relevant evidence found."
                    )

                else:

                    # ============================================
                    # RETRIEVED EVIDENCE
                    # ============================================

                    with st.expander(
                        "🔎 View Retrieved Evidence"
                    ):

                        for item in retrieved_pages:

                            st.markdown(
    f"""
    <div class="evidence-card">

    <div class="source-tag">
    {item["document"]}
    </div>

    <br><br>

    <b>📄 Page {item["page"]}</b>

    <br>

    <span class="small-muted">
    Similarity score:
    {item["score"]:.4f}
    </span>

    </div>
    """,
                                unsafe_allow_html=True
                            )

                            text_preview = item.get(
                                "text",
                                ""
                            )

                            if text_preview:

                                st.write(
                                    text_preview[:1500]
                                )

                            table_preview = item.get(
                                "table_text",
                                ""
                            )

                            if table_preview:

                                st.markdown(
                                    "**📊 Table:**"
                                )

                                st.code(
                                    table_preview[:2500]
                                )

                    # ============================================
                    # GEMINI
                    # ============================================

                    with st.spinner(
                        "🤖 Gemini is analyzing the evidence..."
                    ):

                        answer = generate_answer(
                            question,
                            retrieved_pages
                        )

                    st.session_state.last_answer = (
                        answer
                    )

                    # ============================================
                    # ANSWER
                    # ============================================

                    st.markdown(
                        "## 🧠 Answer"
                    )

                    st.markdown(
    """
    <div class="answer-card">
    """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        answer
                    )

                    st.markdown(
    """
    </div>
    """,
                        unsafe_allow_html=True
                    )

                    # ============================================
                    # SUPPORTING EVIDENCE
                    # ============================================

                    display_supporting_pages(
                        answer,
                        retrieved_pages
                    )



# ============================================================
# FEATURES
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    "## 🎯 Project Capabilities"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
"""
### 📄 Document Intelligence

• PDF text extraction

• Table extraction

• OCR

• Page-level metadata

• Multiple PDF documents
"""
    )

with col2:

    st.markdown(
"""
### 👁️ Multimodal Analysis

• Page images

• Charts

• Graphs

• Tables

• Numerical reasoning
"""
    )

with col3:

    st.markdown(
"""
### 📌 Evidence-Based Answers

• Document citations

• Page citations

• Supporting images

• Cross-document retrieval

• Reduced hallucination
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""
<div style="
text-align:center;
padding:30px;
color:#8993b5;
font-size:13px;
">

🧠 DocMind AI

<br>

Multimodal Document Intelligence &
Evidence-Based Question Answering

</div>
""",
unsafe_allow_html=True
)
