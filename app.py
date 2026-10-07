import streamlit as st
import fitz
import os
import time
import io
import numpy as np
import pandas as pd
import pytesseract

from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocMind AI",
    page_icon="📄",
    layout="wide"
)

MODEL_NAME = "gemini-3.7-flash"

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_REAL_KEY":
    st.error("Gemini API key is missing. Please check your .env file.")
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

# Windows Tesseract installation path
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# SESSION STATE
# ============================================================

if "pages" not in st.session_state:
    st.session_state.pages = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_ocr_text(image_bytes):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # OCR
        text = pytesseract.image_to_string(
            image
        )

        return text.strip()

    except Exception as e:

        return ""


# ============================================================
# PDF PAGE IMAGE FUNCTION
# ============================================================

def render_page_image(page):

    try:

        # Render PDF page
        matrix = fitz.Matrix(1.5, 1.5)

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        image_bytes = pix.tobytes("png")

        return image_bytes

    except Exception:

        return None


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_tables(page):

    tables = []

    try:

        page_tables = page.find_tables()

        for table in page_tables.tables:

            try:

                df = table.to_pandas()

                if not df.empty:

                    tables.append(df)

            except Exception:

                continue

    except Exception:

        pass

    return tables


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    document_pages = []

    try:

        pdf_bytes = uploaded_file.read()

        pdf = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        total_pages = len(pdf)

        progress = st.progress(0)

        status = st.empty()

        for page_number, page in enumerate(pdf):

            status.text(
                f"Processing {uploaded_file.name} "
                f"- Page {page_number + 1}/{total_pages}"
            )

            # ------------------------------------------------
            # TEXT EXTRACTION
            # ------------------------------------------------

            text = page.get_text("text").strip()

            # ------------------------------------------------
            # TABLE EXTRACTION
            # ------------------------------------------------

            tables = extract_tables(page)

            # ------------------------------------------------
            # PAGE IMAGE
            # ------------------------------------------------

            image_bytes = render_page_image(page)

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            ocr_used = False

            # If normal text extraction gives little text,
            # use OCR for scanned/image-only pages.

            if len(text) < 50 and image_bytes is not None:

                ocr_text = extract_ocr_text(
                    image_bytes
                )

                if len(ocr_text) > len(text):

                    text = ocr_text

                    ocr_used = True

            # ------------------------------------------------
            # TABLE TEXT
            # ------------------------------------------------

            table_text = ""

            for table_index, df in enumerate(tables):

                table_text += (
                    f"\nTable {table_index + 1}:\n"
                )

                table_text += df.to_string(
                    index=False
                )

                table_text += "\n"

            # ------------------------------------------------
            # COMBINED SEARCH TEXT
            # ------------------------------------------------

            search_text = (
                text
                + "\n"
                + table_text
            )

            # ------------------------------------------------
            # PAGE INFORMATION
            # ------------------------------------------------

            page_data = {

                "document": uploaded_file.name,

                "page": page_number + 1,

                "text": text,

                "tables": tables,

                "table_text": table_text,

                "search_text": search_text,

                "image": image_bytes,

                "ocr_used": ocr_used

            }

            document_pages.append(
                page_data
            )

            progress.progress(
                (page_number + 1) / total_pages
            )

        status.empty()

        progress.empty()

        pdf.close()

        return document_pages

    except Exception as e:

        st.error(
            f"Error processing {uploaded_file.name}: {e}"
        )

        return []


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(pages):

    if not pages:

        return None

    texts = []

    for page in pages:

        text = page["search_text"]

        if not text.strip():

            text = (
                f"Document {page['document']} "
                f"Page {page['page']}"
            )

        texts.append(text)

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(embeddings)


# ============================================================
# RETRIEVE RELEVANT PAGES
# ============================================================

def retrieve_pages(
    question,
    pages,
    embeddings,
    top_k=8
):

    if not pages or embeddings is None:

        return []

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        embeddings,
        question_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        page = pages[index].copy()

        page["similarity"] = float(
            scores[index]
        )

        results.append(page)

    return results


# ============================================================
# BUILD GEMINI PROMPT
# ============================================================

def build_prompt(question, retrieved_pages):

    evidence_text = ""

    for item in retrieved_pages:

        evidence_text += "\n"
        evidence_text += "=" * 70
        evidence_text += "\n"

        evidence_text += (
            f"DOCUMENT: {item['document']}\n"
        )

        evidence_text += (
            f"PAGE: {item['page']}\n"
        )

        evidence_text += "\nTEXT:\n"

        evidence_text += (
            item["text"][:12000]
        )

        if item["table_text"]:

            evidence_text += (
                "\n\nTABLE DATA:\n"
            )

            evidence_text += (
                item["table_text"][:10000]
            )

        evidence_text += "\n"

    prompt = f"""
You are DocMind AI, a multimodal document
intelligence assistant.

Your job is to answer the user's question ONLY
using the retrieved document evidence provided below.

USER QUESTION:
{question}

RETRIEVED DOCUMENT EVIDENCE:
{evidence_text}

IMPORTANT INSTRUCTIONS:

1. Use ONLY the retrieved documents as factual evidence.

2. Every important factual claim must contain
   an exact citation in this format:

   [Document: filename, Page: number]

3. If information comes from a table, use the table
   information directly.

4. If the question requires numerical calculations,
   perform the calculation carefully.

5. If the question involves a chart, graph, diagram,
   figure, image, scanned page, or visual information,
   inspect the supplied page images.

6. Do not assume that information exists if it is not
   visible or present in the retrieved evidence.

7. If there is not enough information to answer the
   question, say:

   "Insufficient evidence in the retrieved documents."

8. When comparing multiple documents, clearly identify
   which document each piece of evidence comes from.

9. Give a concise but useful explanation.

10. At the end, provide:

Evidence Used:
- Document name, Page number
- Document name, Page number

The final answer should be evidence-based and
traceable to the supplied documents.
"""

    return prompt


# ============================================================
# GEMINI MULTIMODAL ANSWER
# ============================================================

def ask_gemini(
    question,
    retrieved_pages
):

    prompt = build_prompt(
        question,
        retrieved_pages
    )

    contents = []

    # Add text prompt
    contents.append(
        types.Part.from_text(
            text=prompt
        )
    )

    # Add page images
    for item in retrieved_pages:

        image_bytes = item.get("image")

        if image_bytes:

            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                )
            )

    # --------------------------------------------------------
    # Retry Gemini if temporarily unavailable
    # --------------------------------------------------------

    retry_waits = [
        10,
        20,
        30
    ]

    for attempt in range(
        len(retry_waits) + 1
    ):

        try:

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=contents

            )

            return response.text

        except Exception as e:

            error_message = str(e)

            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
                or "temporarily" in error_message.lower()
            ):

                if attempt < len(retry_waits):

                    wait_time = retry_waits[attempt]

                    st.warning(
                        f"Gemini is temporarily busy. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )

                else:

                    return (
                        "Gemini is currently unavailable. "
                        "Please try again later."
                    )

            else:

                return (
                    f"Gemini error: {error_message}"
                )

    return "Unable to generate an answer."


# ============================================================
# APPLICATION UI
# ============================================================

st.title(
    "📄 DocMind AI"
)

st.subheader(
    "Multimodal Document Intelligence "
    "and Evidence-Based Question Answering"
)

st.write(
    "Upload multiple PDFs and ask questions "
    "using text, tables, scanned pages, and "
    "document images."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved pages",
        min_value=3,
        max_value=8,
        value=8
    )

    st.markdown("---")

    st.write(
        "### Supported Features"
    )

    st.write(
        "📄 PDF documents"
    )

    st.write(
        "📝 Text extraction"
    )

    st.write(
        "📊 Table extraction"
    )

    st.write(
        "🔤 OCR for scanned pages"
    )

    st.write(
        "🖼️ Page image analysis"
    )

    st.write(
        "🔎 Semantic search"
    )

    st.write(
        "🤖 Gemini multimodal reasoning"
    )

    st.write(
        "📌 Page-level evidence"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_files = st.file_uploader(

    "Upload PDF documents",

    type=["pdf"],

    accept_multiple_files=True
)


# ============================================================
# PROCESS UPLOADED DOCUMENTS
# ============================================================

if uploaded_files:

    st.session_state.pages = []

    st.session_state.embeddings = None

    st.subheader(
        "📚 Loaded Documents"
    )

    for uploaded_file in uploaded_files:

        with st.spinner(
            f"Processing {uploaded_file.name}..."
        ):

            pages = process_pdf(
                uploaded_file
            )

        st.session_state.pages.extend(
            pages
        )

        st.success(
            f"{uploaded_file.name} "
            f"— {len(pages)} pages processed"
        )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    with st.spinner(
        "Creating semantic embeddings..."
    ):

        st.session_state.embeddings = (
            create_embeddings(
                st.session_state.pages
            )
        )

    st.success(
        f"Processed {len(st.session_state.pages)} pages successfully."
    )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.pages:

    st.subheader(
        "📑 Document Summary"
    )

    document_names = sorted(
        set(
            page["document"]
            for page in st.session_state.pages
        )
    )

    for document_name in document_names:

        document_pages = [
            page
            for page in st.session_state.pages
            if page["document"] == document_name
        ]

        ocr_pages = sum(
            page["ocr_used"]
            for page in document_pages
        )

        table_pages = sum(
            len(page["tables"]) > 0
            for page in document_pages
        )

        st.write(
            f"**{document_name}** — "
            f"{len(document_pages)} pages | "
            f"OCR pages: {ocr_pages} | "
            f"Pages containing tables: {table_pages}"
        )


# ============================================================
# QUESTION ANSWERING
# ============================================================

if st.session_state.pages:

    st.markdown("---")

    st.subheader(
        "🔎 Ask a Question"
    )

    question = st.text_area(
        "Enter your question:",
        placeholder=(
            "Example: Compare the production "
            "efficiency between Q2 and Q4. "
            "Identify the three biggest reasons "
            "for the change and provide the evidence."
        ),
        height=100
    )

    ask_button = st.button(
        "🚀 Ask DocMind AI",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------------
            # RETRIEVAL
            # ------------------------------------------------

            with st.spinner(
                "Searching documents..."
            ):

                retrieved_pages = retrieve_pages(

                    question,

                    st.session_state.pages,

                    st.session_state.embeddings,

                    top_k

                )

            # ------------------------------------------------
            # RETRIEVED EVIDENCE
            # ------------------------------------------------

            st.subheader(
                "🔍 Retrieved Evidence"
            )

            for item in retrieved_pages:

                st.write(
                    f"**{item['document']}** "
                    f"— Page {item['page']} "
                    f"— Similarity: "
                    f"{item['similarity']:.3f}"
                )

                if item["ocr_used"]:

                    st.caption(
                        "🔤 OCR was used on this page."
                    )

            # ------------------------------------------------
            # GEMINI ANSWER
            # ------------------------------------------------

            st.subheader(
                "🤖 DocMind AI Answer"
            )

            with st.spinner(
                "Analyzing evidence and generating answer..."
            ):

                answer = ask_gemini(

                    question,

                    retrieved_pages

                )

            st.markdown(
                answer
            )

            # ------------------------------------------------
            # SUPPORTING PAGE IMAGES
            # ------------------------------------------------

            st.subheader(
                "🖼️ Supporting Evidence Pages"
            )

            for item in retrieved_pages:

                image_bytes = item.get(
                    "image"
                )

                if image_bytes:

                    st.markdown(
                        f"**{item['document']} "
                        f"— Page {item['page']}**"
                    )

                    st.image(
                        image_bytes,
                        use_container_width=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "DocMind AI | Multimodal Document Intelligence | "
    "Text + Tables + OCR + Page Images + RAG + Gemini"
)