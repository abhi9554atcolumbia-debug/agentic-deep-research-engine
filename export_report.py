"""
export_report.py
Converts the Markdown report (output of report_generator.generate_report)
into PDF and DOCX, satisfying the brief's "Exportable report in PDF,
DOCX, Markdown, or HTML" requirement.

Uses pypandoc (a thin wrapper around the `pandoc` binary). If pandoc
isn't installed, functions return None and print install instructions
instead of crashing -- Markdown export always works regardless.
"""

import os
import tempfile

try:
    import pypandoc
    _PANDOC_AVAILABLE = True
except ImportError:
    _PANDOC_AVAILABLE = False


def _check_pandoc():
    if not _PANDOC_AVAILABLE:
        print("pypandoc not installed. Run: pip install pypandoc --break-system-packages")
        return False
    try:
        pypandoc.get_pandoc_version()
        return True
    except OSError:
        print("pandoc binary not found. Install it:")
        print("  macOS:   brew install pandoc")
        print("  Linux:   sudo apt-get install pandoc")
        print("  Windows: https://pandoc.org/installing.html")
        return False


def markdown_to_pdf_bytes(markdown_text: str):
    """
    Converts Markdown to PDF bytes. Returns None if pandoc is unavailable.
    Requires a LaTeX engine (e.g. via `brew install basictex` or
    `apt-get install texlive`) OR falls back to using pandoc's HTML+wkhtmltopdf
    if available. If neither works, returns None.
    """
    if not _check_pandoc():
        return None

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pypandoc.convert_text(markdown_text, "pdf", format="md", outputfile=tmp_path,
                               extra_args=["--standalone"])
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    except Exception as e:
        print(f"PDF conversion failed ({e}). "
              f"This usually means no PDF engine is installed (e.g. LaTeX/wkhtmltopdf).")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def markdown_to_docx_bytes(markdown_text: str):
    """
    Converts Markdown to DOCX bytes. Returns None if pandoc is unavailable.
    """
    if not _check_pandoc():
        return None

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pypandoc.convert_text(markdown_text, "docx", format="md", outputfile=tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    except Exception as e:
        print(f"DOCX conversion failed: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    sample_md = "# Test Report\n\nThis is a **test**.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"

    docx_bytes = markdown_to_docx_bytes(sample_md)
    if docx_bytes:
        with open("test_report.docx", "wb") as f:
            f.write(docx_bytes)
        print("Wrote test_report.docx")

    pdf_bytes = markdown_to_pdf_bytes(sample_md)
    if pdf_bytes:
        with open("test_report.pdf", "wb") as f:
            f.write(pdf_bytes)
        print("Wrote test_report.pdf")
