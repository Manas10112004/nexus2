import streamlit as st
from fpdf import FPDF
import os


class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Nexus AI - Session Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_pdf(history, session_id):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)

    pdf.cell(0, 10, f"Session ID: {session_id}", ln=True)
    pdf.ln(5)

    for msg in history:
        role = msg["role"].upper()
        content = msg["content"]

        # Clean text to prevent latin-1 encoding errors
        content = content.encode('latin-1', 'replace').decode('latin-1')

        # Role Header
        pdf.set_font("Arial", 'B', 10)
        # Blue for User, Green for AI
        pdf.set_text_color(0, 50, 150) if role == "USER" else pdf.set_text_color(0, 100, 50)
        pdf.cell(0, 6, f"[{role}]", ln=True)

        # Content
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, content)
        pdf.ln(3)

    # --- FIX: Look for the specific session chart in the current directory ---
    chart_filename = f"chart_{session_id}.png"

    # Check current directory
    if os.path.exists(chart_filename):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Attached Analysis Chart:", ln=True)

        # Image placement (x, y, width)
        # We constrain width to 180 to fit page
        pdf.image(chart_filename, x=10, y=30, w=180)
    else:
        # Debug line (Optional: helps you see if it failed silently)
        # pdf.cell(0, 10, f"[Debug] Chart file not found: {chart_filename}", ln=True)
        pass

    # Output PDF
    output_filename = f"report_{session_id}.pdf"
    pdf.output(output_filename)
    return output_filename