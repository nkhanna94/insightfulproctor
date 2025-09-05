from fpdf import FPDF
import io
import os
from PIL import Image
import cv2

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        self.add_font("NotoSans", "", os.path.join("fonts", "NotoSans-Regular.ttf"), uni=True)
        self.set_font("NotoSans", "", 14)

def generate_pdf_report(summaries, frames):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.cell(0, 10, "📄 Proctoring Report", ln=True)
    pdf.ln(5)

    summaries = [s for s in summaries if s["card"] != "Green 🟢"] # this makes sure that we are showing only the violated frames in the pdf

    for summary in summaries:
        index = summary["index"]
        score = summary["score"]
        card = summary["card"]
        caption = summary["caption"]
        violations = "".join(f"{v}" for v in summary.get("violations", []))
        frame_img_cv = frames[index+1]

        pdf.set_font("NotoSans", "", 12)
        pdf.cell(0, 10, f"📸 Frame {index+1} - Score: {score} - Card: {card}", ln=True)

        img = Image.fromarray(cv2.cvtColor(frame_img_cv, cv2.COLOR_BGR2RGB))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        img_path = f"temp_frame_{index}.png"
        img.save(img_path)
        pdf.image(img_path, w=100)
        os.remove(img_path)

        pdf.multi_cell(0, 8, caption)
        pdf.ln(2)

        pdf.set_font("NotoSans", "", 12)
        pdf.multi_cell(0, 10, "Violations:", ln=True)
        pdf.multi_cell(180, 8, violations)

        pdf.ln(5)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output.getvalue()