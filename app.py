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
                #0b1020 35%,
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

    .hero {
        padding: 35px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(83, 102, 255, 0.28),
                rgba(18, 25, 55, 0.90)
            );
        border: 1px solid rgba(130, 145, 255, 0.25);
        margin-bottom: 30px;
        box-shadow: 0 15px 45px rgba(0,0,0,0.20);
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

    .pipeline {
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 25px;
    }

    .pipeline-item {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
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

    .section-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px;
        margin-top: 15px;
    }

    .evidence-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .source-tag {
        display: inline-block;
        background: rgba(102, 118, 255, 0.18);
        border: 1px solid rgba(102, 118, 255, 0.30);
        padding: 5px 10px;
        border-radius: 10px;
        color: #dbe0ff;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .small-muted {
        color: #9fa9cb;
        font-size: 13px;
    }

    .success-box {
        padding: 15px;
        border-radius: 14px;
        background: rgba(40, 180, 110, 0.10);
        border: 1px solid rgba(40, 180, 110, 0.25);
    }

    .warning-box {
        padding: 15px;
        border-radius: 14px;
        background: rgba(255, 180, 50, 0.10);
        border: 1px solid rgba(255, 180, 50, 0.25);
    }

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
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY or API_KEY == "YOUR_REAL_KEY":
    st.error(
        "Gemini API key is not configured. "
        "Please add GEMINI_API_KEY to your .env file."
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Unable to initialize Gemini: {e}")
    st.stop()


PRIMARY_MODEL = "gemini-3.5-flash"
BACKUP_MODEL = "gemini-3.7-flash"


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


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


# ============================================================
# LOAD EMBEDDING MODEL
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
            1.3,
            1.3
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
# TABLE EXTRACTION
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

    for i, df in enumerate(tables):

        output.append(
            f"TABLE {i + 1}"
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

        # --------------------------------
        # TEXT
        # --------------------------------

        try:

            text = page.get_text(
                "text"
            ).strip()

        except Exception:

            text = ""

        # --------------------------------
        # TABLES
        # --------------------------------

        tables = extract_tables(
            page
        )

        table_text = tables_to_text(
            tables
        )

        # --------------------------------
        # IMAGE
        # --------------------------------

        image = render_page(
            page
        )

        # --------------------------------
        # OCR
        # --------------------------------

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

        # --------------------------------
        # SEARCH TEXT
        # --------------------------------

        search_text = (
            text
            + "\n"
            + table_text
        )

        page_data = {

            "document": document_name,

            "page": page_number + 1,

            "text": text,

            "tables": tables,

            "table_text": table_text,

            "search_text": search_text,

            "image": image,

            "ocr_used": ocr_used
        }

        document_pages.append(
            page_data
        )

    document.close()

    return document_pages


# ============================================================
# EMBEDDINGS
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
                f"Document: "
                f"{page['document']} "
                f"Page: "
                f"{page['page']}"
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

        page = pages[index].copy()

        page["score"] = float(
            scores[index]
        )

        results.append(
            page
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
                + text[:12000]
            )

        if table_text:

            section += (
                "\n\nTABLE CONTENT:\n"
                + table_text[:12000]
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
You are DocMind AI, a multimodal document intelligence system.

Answer the user's question using ONLY the evidence provided from
the uploaded documents.

USER QUESTION:
{question}

RETRIEVED DOCUMENT EVIDENCE:
{evidence}

RULES:

1. Use only information supported by the documents.

2. Never invent facts, numbers, percentages, dates, names,
or conclusions.

3. Every important factual statement must include a citation:

[Document: filename.pdf, Page: 2]

4. If information comes from a table, cite the page containing
the table.

5. If a comparison is requested, explicitly compare the values.

6. If calculations are required, show the calculation.

7. If charts, graphs, or images are visible in the page images,
analyze the visual information.

8. If the evidence is insufficient, say:

Insufficient evidence in the retrieved documents.

9. If the user asks for multiple reasons, use a numbered list.

10. Do not cite pages that do not support your answer.

11. Keep the answer clear and technically accurate.

12. At the end, provide:

CITATIONS:
- filename.pdf | Page 2
- filename.pdf | Page 5

Only include pages that actually support the answer.

Now answer the question.
"""


# ============================================================
# GEMINI ANSWER
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

    contents = [
        types.Part.from_text(
            text=prompt
        )
    ]

    # --------------------------------
    # Add page images
    # --------------------------------

    for item in retrieved_pages[:3]:

        image = item.get(
            "image"
        )

        if image is None:

            continue

        try:

            image_buffer = io.BytesIO()

            image.save(
                image_buffer,
                format="PNG"
            )

            contents.append(
                types.Part.from_bytes(
                    data=image_buffer.getvalue(),
                    mime_type="image/png"
                )
            )

        except Exception:

            continue

    # --------------------------------
    # Try models
    # --------------------------------

    models = [
        PRIMARY_MODEL,
        BACKUP_MODEL
    ]

    last_error = None

    for model_name in models:

        try:

            response = (
                client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
            )

            if (
                response
                and response.text
            ):

                return response.text

        except Exception as e:

            last_error = e

            time.sleep(1)

            continue

    if last_error:

        return (
            "Unable to generate the answer right now.\n\n"
            f"Gemini error: {last_error}\n\n"
            "The document retrieval system is working, "
            "but the Gemini generation service is currently unavailable."
        )

    return "Unable to generate an answer."


# ============================================================
# PARSE CITATIONS
# ============================================================

def parse_citations(answer):

    citations = []

    if not answer:

        return citations

    # Structured citations

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
# SUPPORTING PAGE IMAGES
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

            same_document = (
                item["document"].strip().lower()
                ==
                citation["document"].strip().lower()
            )

            same_page = (
                item["page"]
                ==
                citation["page"]
            )

            if (
                same_document
                and same_page
            ):

                if item not in cited_pages:

                    cited_pages.append(
                        item
                    )

    # Fallback

    if not cited_pages:

        cited_pages = retrieved_pages[:3]

    if not cited_pages:

        st.info(
            "No supporting page images available."
        )

        return

    st.markdown(
        "### 🖼️ Supporting Evidence Pages"
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

                    <br>

                    <b>
                        📄 Page {item["page"]}
                    </b>

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
        and generates answers with
        document and page citations.
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

        **5. Page Image**

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
        "### 🤖 AI Configuration"
    )

    st.caption(
        f"Primary: {PRIMARY_MODEL}"
    )

    st.caption(
        f"Backup: {BACKUP_MODEL}"
    )

    st.divider()

    st.markdown(
        "### 🔐 Security"
    )

    st.caption(
        "API keys are loaded from the .env file."
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">🧠 DocMind AI</div>

<div class="hero-subtitle">
Multimodal Document Intelligence &amp;
Evidence-Based Question Answering
</div>

<div class="pipeline">

<div class="pipeline-item">📄 Documents</div>

<div class="arrow">→</div>

<div class="pipeline-item">🔎 Extract</div>

<div class="arrow">→</div>

<div class="pipeline-item">🧩 Retrieve</div>

<div class="arrow">→</div>

<div class="pipeline-item">👁️ Analyze</div>

<div class="arrow">→</div>

<div class="pipeline-item">📌 Cite</div>

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
    "Upload one or more PDF documents to begin."
)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    st.markdown(
        f"""
        <div class="success-box">
            <b>✅ {len(uploaded_files)} PDF file(s) selected.</b>
            Click the button below to process them.
        </div>
        """,
        unsafe_allow_html=True
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

        progress = st.progress(
            0
        )

        status = st.empty()

        all_pages = []

        total_files = len(
            uploaded_files
        )

        for file_index, uploaded_file in enumerate(
            uploaded_files
        ):

            status.write(
                f"📄 Processing {uploaded_file.name}..."
            )

            try:

                pages = process_pdf(
                    uploaded_file
                )

                all_pages.extend(
                    pages
                )

                st.session_state.documents.append(
                    uploaded_file.name
                )

            except Exception as e:

                st.error(
                    f"Error processing {uploaded_file.name}: {e}"
                )

            progress.progress(
                (file_index + 1)
                / total_files
            )

        if all_pages:

            status.write(
                "🧩 Creating semantic embeddings..."
            )

            embeddings = create_embeddings(
                all_pages
            )

            st.session_state.pages = (
                all_pages
            )

            st.session_state.embeddings = (
                embeddings
            )

            st.session_state.processed = True

            status.success(
                "✅ Documents processed successfully."
            )

        else:

            st.error(
                "No pages were extracted."
            )


# ============================================================
# WORKSPACE STATISTICS
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

    ocr_pages = sum(
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

    c1, c2, c3, c4 = st.columns(4)

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
                    {ocr_pages}
                </div>

                <div class="stat-label">
                    OCR Pages
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
# QUESTION ANSWERING
# ============================================================

if st.session_state.processed:

    st.markdown(
        "## 💬 Ask Your Documents"
    )

    st.write(
        "Ask questions about text, tables, "
        "numbers, charts, or multiple documents."
    )

    question = st.text_area(
        "Enter your question",
        height=130,
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

            # --------------------------------
            # RETRIEVAL
            # --------------------------------

            with st.spinner(
                "🔎 Searching relevant document pages..."
            ):

                retrieved_pages = retrieve_pages(
                    question,
                    st.session_state.pages,
                    st.session_state.embeddings,
                    top_k=4
                )

            if not retrieved_pages:

                st.error(
                    "No relevant evidence was found."
                )

            else:

                # --------------------------------
                # SHOW RETRIEVED EVIDENCE
                # --------------------------------

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

                                <br>

                                <b>
                                    Page {item["page"]}
                                </b>

                                <br>

                                <span class="small-muted">
                                    Retrieval score:
                                    {item["score"]:.4f}
                                </span>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        preview_text = item.get(
                            "text",
                            ""
                        )

                        if preview_text:

                            st.write(
                                preview_text[:1200]
                            )

                        table_text = item.get(
                            "table_text",
                            ""
                        )

                        if table_text:

                            st.markdown(
                                "**Table Content:**"
                            )

                            st.code(
                                table_text[:2000]
                            )

                # --------------------------------
                # GENERATE ANSWER
                # --------------------------------

                with st.spinner(
                    "🤖 Gemini is analyzing the documents..."
                ):

                    answer = generate_answer(
                        question,
                        retrieved_pages
                    )

                st.session_state.last_answer = (
                    answer
                )

                # --------------------------------
                # DISPLAY ANSWER
                # --------------------------------

                st.markdown(
                    "## 🧠 Answer"
                )

                st.markdown(
                    '<div class="answer-card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    answer
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                # --------------------------------
                # SUPPORTING PAGES
                # --------------------------------

                display_supporting_pages(
                    answer,
                    retrieved_pages
                )


# ============================================================
# PROJECT FEATURES
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    "## 🎯 Project Capabilities"
)

cap1, cap2, cap3 = st.columns(3)

with cap1:

    st.markdown(
        """
        ### 📄 Document Intelligence

        • PDF text extraction

        • Table extraction

        • OCR for scanned pages

        • Page-level metadata

        • Multiple PDF support
        """
    )

with cap2:

    st.markdown(
        """
        ### 👁️ Multimodal Analysis

        • Page image analysis

        • Chart understanding

        • Table reasoning

        • Numerical comparison

        • Cross-document retrieval
        """
    )

with cap3:

    st.markdown(
        """
        ### 📌 Evidence & Citations

        • Document-level citations

        • Page-level citations

        • Supporting page images

        • Evidence-based answers

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