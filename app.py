import streamlit as st
import fitz
import os
import io
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
    page_icon="📄",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("📄 DocMind AI")

st.subheader(
    "Multimodal Document Intelligence and Evidence-Based Question Answering"
)

st.write(
    """
    Upload one or more PDF documents and ask questions about their
    text, tables, scanned pages, charts, graphs and images.
    
    DocMind AI retrieves relevant evidence and uses a Vision-Language
    Model to generate answers with document and page-level citations.
    """
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    st.error(
        "Gemini API key not found. Please check your .env file."
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

try:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

except Exception as e:

    st.error(
        f"Unable to initialize Gemini client: {e}"
    )

    st.stop()


# ============================================================
# GEMINI MODELS
# ============================================================

# Primary model
PRIMARY_MODEL = "gemini-3.6-flash"

# Backup model
BACKUP_MODEL = "gemini-3.5-flash"


# ============================================================
# TESSERACT OCR CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# OCR FUNCTION
# ============================================================

def perform_ocr(image):

    try:

        text = pytesseract.image_to_string(
            image
        )

        return text.strip()

    except Exception as e:

        return ""


# ============================================================
# RENDER PDF PAGE AS IMAGE
# ============================================================

def render_page(page):

    try:

        matrix = fitz.Matrix(
            1.5,
            1.5
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
        )

        return image

    except Exception:

        return None


# ============================================================
# EXTRACT TABLES
# ============================================================

def extract_tables(page):

    tables = []

    try:

        table_finder = page.find_tables()

        for table in table_finder.tables:

            try:

                dataframe = table.to_pandas()

                if dataframe is not None:

                    tables.append(
                        dataframe
                    )

            except Exception:

                continue

    except Exception:

        pass

    return tables


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(uploaded_file):

    document_name = uploaded_file.name

    pdf_bytes = uploaded_file.read()

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_index in range(
        len(pdf_document)
    ):

        page_number = page_index + 1

        page = pdf_document[
            page_index
        ]

        # ----------------------------------------------------
        # Extract normal PDF text
        # ----------------------------------------------------

        extracted_text = page.get_text(
            "text"
        ).strip()


        # ----------------------------------------------------
        # Render page
        # ----------------------------------------------------

        page_image = render_page(
            page
        )


        # ----------------------------------------------------
        # OCR if page contains little/no text
        # ----------------------------------------------------

        ocr_used = False

        ocr_text = ""

        if (
            len(extracted_text) < 50
            and page_image is not None
        ):

            ocr_text = perform_ocr(
                page_image
            )

            if len(ocr_text) > len(
                extracted_text
            ):

                extracted_text = ocr_text

                ocr_used = True


        # ----------------------------------------------------
        # Extract tables
        # ----------------------------------------------------

        tables = extract_tables(
            page
        )


        # ----------------------------------------------------
        # Convert tables to text
        # ----------------------------------------------------

        table_text_parts = []

        for table_number, dataframe in enumerate(
            tables,
            start=1
        ):

            try:

                table_text = dataframe.to_string(
                    index=False
                )

                table_text_parts.append(
                    f"Table {table_number}:\n{table_text}"
                )

            except Exception:

                continue


        table_text = "\n\n".join(
            table_text_parts
        )


        # ----------------------------------------------------
        # Unified searchable content
        # ----------------------------------------------------

        search_text = extracted_text

        if table_text:

            search_text += (
                "\n\n"
                + table_text
            )


        # ----------------------------------------------------
        # Save page information
        # ----------------------------------------------------

        pages.append(
            {
                "document": document_name,
                "page": page_number,
                "text": extracted_text,
                "tables": tables,
                "table_text": table_text,
                "search_text": search_text,
                "image": page_image,
                "ocr_used": ocr_used
            }
        )


    pdf_document.close()

    return pages


# ============================================================
# PROCESS ALL DOCUMENTS
# ============================================================

def process_documents(uploaded_files):

    all_pages = []

    progress_bar = st.progress(
        0
    )

    total_files = len(
        uploaded_files
    )

    for file_index, uploaded_file in enumerate(
        uploaded_files
    ):

        with st.spinner(
            f"Processing {uploaded_file.name}..."
        ):

            try:

                pages = process_pdf(
                    uploaded_file
                )

                all_pages.extend(
                    pages
                )

            except Exception as e:

                st.error(
                    f"Error processing {uploaded_file.name}: {e}"
                )


        progress_bar.progress(
            (file_index + 1)
            / total_files
        )


    progress_bar.empty()

    return all_pages


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(pages):

    if not pages:

        return np.array([])


    texts = []

    for page in pages:

        text = page.get(
            "search_text",
            ""
        )

        if not text.strip():

            text = (
                f"Document: {page['document']} "
                f"Page: {page['page']}"
            )

        texts.append(
            text
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
# RETRIEVE RELEVANT PAGES
# ============================================================

def retrieve_relevant_pages(
    question,
    pages,
    embeddings,
    top_k=6
):

    if (
        not pages
        or embeddings.size == 0
    ):

        return []


    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]


    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarities = np.dot(
        embeddings,
        query_embedding
    )


    # --------------------------------------------------------
    # Get top results
    # --------------------------------------------------------

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]


    results = []

    for index in top_indices:

        result = pages[index].copy()

        result["similarity"] = float(
            similarities[index]
        )

        results.append(
            result
        )


    return results


# ============================================================
# FORMAT EVIDENCE
# ============================================================

def build_evidence_text(retrieved_pages):

    evidence_sections = []

    for item in retrieved_pages:

        document = item["document"]
        page = item["page"]

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

        evidence_sections.append(section)

    return "\n\n" + (
        "\n\n" + "=" * 70 + "\n\n"
    ).join(evidence_sections)


# ============================================================
# CREATE GEMINI PROMPT
# ============================================================

def create_prompt(
    question,
    retrieved_pages
):

    evidence = build_evidence_text(
        retrieved_pages
    )


    prompt = f"""
You are DocMind AI, a multimodal document intelligence assistant.

Your task is to answer the user's question using ONLY the evidence
provided from the uploaded documents and the page images.

USER QUESTION:
{question}

RETRIEVED DOCUMENT EVIDENCE:
{evidence}

IMPORTANT INSTRUCTIONS:

1. Answer the user's question directly.

2. Use the retrieved text and tables as evidence.

3. Carefully inspect the supplied page images.

4. Page images may contain:
   - charts
   - graphs
   - scanned text
   - diagrams
   - images
   - tables
   - visual information that may not appear in extracted text

5. If the question asks about a chart or graph, analyze the actual
   visual chart rather than assuming the answer from nearby text.

6. If the question asks for numbers, calculate them carefully.

7. For comparisons, clearly identify the values being compared.

8. For cross-document questions, compare information from the
   relevant documents.

9. EVERY important factual statement must include a citation in this
   exact format:

   [Document: filename, Page: number]

10. Do not invent document names or page numbers.

11. If the evidence does not contain enough information, say:

   "Insufficient evidence in the retrieved documents."

12. Do not use outside knowledge.

13. If calculations are required, show the calculation briefly.

14. At the end, provide:

   Evidence Used:
   - [Document: filename, Page: number]
   - [Document: filename, Page: number]

15. Keep the answer clear and easy to understand.

Return a professional evidence-based answer.
"""

    return prompt


# ============================================================
# GENERATE GEMINI ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_pages
):

    prompt = create_prompt(
        question,
        retrieved_pages
    )


    # --------------------------------------------------------
    # Build Gemini content
    # --------------------------------------------------------

    contents = [
        prompt
    ]


    # --------------------------------------------------------
    # Add page images
    # --------------------------------------------------------

    for item in retrieved_pages:

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

            image_bytes = (
                image_buffer.getvalue()
            )


            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            )


            contents.append(
                image_part
            )

        except Exception:

            continue


    # --------------------------------------------------------
    # Models to try
    # --------------------------------------------------------

    models_to_try = [
        PRIMARY_MODEL,
        BACKUP_MODEL
    ]


    last_error = None


    # --------------------------------------------------------
    # Try each model
    # --------------------------------------------------------

    for model_name in models_to_try:

        for attempt in range(2):

            try:

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )


                if response is not None:

                    answer = response.text

                    if answer:

                        return answer


            except Exception as e:

                last_error = e

                error_message = str(
                    e
                ).lower()


                # ------------------------------------------------
                # Retry temporary errors
                # ------------------------------------------------

                temporary_error = (
                    "503" in error_message
                    or "unavailable" in error_message
                    or "temporarily" in error_message
                    or "overloaded" in error_message
                    or "busy" in error_message
                    or "resource exhausted" in error_message
                    or "429" in error_message
                )


                if temporary_error:

                    if attempt == 0:

                        st.warning(
                            f"{model_name} is temporarily busy. "
                            f"Retrying..."
                        )

                        time.sleep(
                            8
                        )

                    continue


                # ------------------------------------------------
                # Other error
                # ------------------------------------------------

                break


    # --------------------------------------------------------
    # All models failed
    # --------------------------------------------------------

    return (
        "Unable to generate the answer right now.\n\n"
        f"Gemini error: {last_error}\n\n"
        "The document retrieval system is working, but the "
        "Gemini generation service is currently unavailable."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )


    top_k = st.slider(
        "Number of pages to retrieve",
        min_value=3,
        max_value=8,
        value=6
    )


    st.divider()


    st.markdown(
        """
        ### Supported Content

        📄 PDF Text  
        📊 Tables  
        📈 Charts  
        📉 Graphs  
        🖼️ Images  
        🔍 OCR Scanned Pages  
        🤖 Vision-Language Analysis  
        🔗 Page-Level Evidence
        """
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header(
    "📂 Upload Documents"
)


uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "pages" not in st.session_state:

    st.session_state.pages = []


if "embeddings" not in st.session_state:

    st.session_state.embeddings = np.array([])


if "processed" not in st.session_state:

    st.session_state.processed = False


# ============================================================
# PROCESS BUTTON
# ============================================================

if uploaded_files:

    if st.button(
        "🔄 Process Documents",
        type="primary"
    ):

        with st.spinner(
            "Processing documents..."
        ):

            pages = process_documents(
                uploaded_files
            )


            embeddings = create_embeddings(
                pages
            )


            st.session_state.pages = pages

            st.session_state.embeddings = embeddings

            st.session_state.processed = True


        st.success(
            f"Successfully processed "
            f"{len(uploaded_files)} document(s) "
            f"and {len(pages)} page(s)."
        )


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

if st.session_state.processed:

    pages = st.session_state.pages


    st.header(
        "📚 Loaded Documents"
    )


    documents = sorted(
        list(
            set(
                page["document"]
                for page in pages
            )
        )
    )


    for document in documents:

        document_pages = [
            page
            for page in pages
            if page["document"] == document
        ]


        ocr_pages = sum(
            1
            for page in document_pages
            if page["ocr_used"]
        )


        table_pages = sum(
            1
            for page in document_pages
            if page["tables"]
        )


        with st.expander(
            f"📄 {document}"
        ):

            st.write(
                f"Pages: {len(document_pages)}"
            )

            st.write(
                f"OCR pages: {ocr_pages}"
            )

            st.write(
                f"Pages containing tables: {table_pages}"
            )


# ============================================================
# QUESTION SECTION
# ============================================================

if st.session_state.processed:

    st.divider()


    st.header(
        "💬 Ask Questions"
    )


    question = st.text_area(
        "Ask a question about your documents:",
        placeholder=(
            "Example: Compare production efficiency "
            "between Q2 and Q4 and explain the reasons."
        ),
        height=100
    )


    if st.button(
        "🔍 Ask DocMind AI",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------------
            # Retrieve evidence
            # ------------------------------------------------

            with st.spinner(
                "Searching documents..."
            ):

                retrieved_pages = retrieve_relevant_pages(
                    question,
                    st.session_state.pages,
                    st.session_state.embeddings,
                    top_k
                )


            # ------------------------------------------------
            # Display retrieved evidence
            # ------------------------------------------------

            st.subheader(
                "🔍 Retrieved Evidence"
            )


            for item in retrieved_pages:

                similarity = item[
                    "similarity"
                ]


                st.write(
                    f"**{item['document']}** "
                    f"— Page **{item['page']}** "
                    f"— Similarity: "
                    f"**{similarity:.3f}**"
                )


            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            st.subheader(
                "🤖 DocMind AI Answer"
            )


            with st.spinner(
                "Analyzing evidence and generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_pages
                )


            st.markdown(
                answer
            )


            # ------------------------------------------------
            # Supporting pages
            # ------------------------------------------------

            st.divider()


            st.subheader(
                "🖼️ Supporting Document Pages"
            )


            for item in retrieved_pages:

                image = item.get(
                    "image"
                )


                if image is None:

                    continue


                caption = (
                    f"{item['document']} — "
                    f"Page {item['page']}"
                )


                st.image(
                    image,
                    caption=caption,
                    width="stretch"
                )


# ============================================================
# INITIAL INSTRUCTIONS
# ============================================================

else:

    st.info(
        """
        👆 Upload one or more PDF documents and click
        **Process Documents** to begin.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocMind AI — Multimodal Document Intelligence "
    "and Evidence-Based Question Answering System"
)
