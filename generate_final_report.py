r"""
Calibrated final report generator compressed to target 35-40 pages
(matching the exact layout, font size, density, and 39-page structure of the sample PDF).
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# Palette matching sample PDF
COLOR_PRIMARY = RGBColor(0x1F, 0x4E, 0x78)    # Deep Navy Blue (#1F4E78)
COLOR_SECONDARY = RGBColor(0x2B, 0x54, 0x7E)  # Slate Blue (#2B547E)
COLOR_TEXT = RGBColor(0x00, 0x00, 0x00)       # Black
COLOR_MUTED = RGBColor(0x55, 0x55, 0x55)      # Muted Gray
COLOR_CODE = RGBColor(0x1E, 0x29, 0x3B)       # Monospace Code Slate

HEX_PRIMARY = "1F4E78"
HEX_ZEBRA = "F2F5F9"
HEX_BORDER = "D0D5DD"
HEX_CODE_BG = "F8F9FA"

def clean_math(text):
    replacements = [
        (r'\pm', '±'),
        (r'\ge', '≥'),
        (r'\le', '≤'),
        (r'\lambda', 'λ'),
        (r'\rho', 'ρ'),
        (r'\chi^2', 'χ²'),
        (r'\chi', 'χ'),
        (r'\approx', '≈'),
        (r'\times', '×'),
        (r'\sum', '∑'),
        (r'\mu', 'μ'),
        (r'\sigma', 'σ'),
        (r'\alpha', 'α'),
        (r'\beta', 'β'),
        (r'\epsilon', 'ε'),
        (r'\text{Holdout}', 'Holdout'),
        (r'\text{Train}', 'Train'),
        (r'\text{Test}', 'Test'),
        (r'\text{Valid}', 'Valid'),
        (r'\text', ''),
        (r'\{', '{'),
        (r'\}', '}'),
        (r'\\', ' '),
        (r'\%', '%'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def parse_inline_tokens(text):
    def math_sub(m):
        return clean_math(m.group(1))
    text = re.sub(r'\$([^\$]+)\$', math_sub, text)

    pattern = re.compile(
        r'(\*\*[^*]+\*\*|'          # **bold**
        r'`[^`]+`|'                 # `code`
        r'\[([^\]]+)\]\(([^)]+)\)|' # [text](url)
        r'\*[^*]+\*|'               # *italic*
        r'_[^_]+_)'                 # _italic_
    )
    tokens = []
    last_idx = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > last_idx:
            tokens.append({'text': text[last_idx:start], 'bold': False, 'italic': False, 'code': False})
        matched_str = match.group(0)
        if matched_str.startswith('**') and matched_str.endswith('**'):
            tokens.append({'text': matched_str[2:-2], 'bold': True, 'italic': False, 'code': False})
        elif matched_str.startswith('`') and matched_str.endswith('`'):
            tokens.append({'text': matched_str[1:-1], 'bold': False, 'italic': False, 'code': True})
        elif matched_str.startswith('[') and '](' in matched_str and matched_str.endswith(')'):
            link_m = re.match(r'\[([^\]]+)\]\(([^)]+)\)', matched_str)
            if link_m:
                tokens.append({'text': link_m.group(1), 'bold': False, 'italic': False, 'code': False, 'link': link_m.group(2)})
            else:
                tokens.append({'text': matched_str, 'bold': False, 'italic': False, 'code': False})
        elif (matched_str.startswith('*') and matched_str.endswith('*')) or (matched_str.startswith('_') and matched_str.endswith('_')):
            tokens.append({'text': matched_str[1:-1], 'bold': False, 'italic': True, 'code': False})
        last_idx = end
    if last_idx < len(text):
        tokens.append({'text': text[last_idx:], 'bold': False, 'italic': False, 'code': False})
    return tokens

def add_styled_runs(paragraph, text, base_font_size=Pt(10), base_color=COLOR_TEXT, default_bold=False, default_italic=False):
    tokens = parse_inline_tokens(text)
    for t in tokens:
        run = paragraph.add_run(t['text'])
        run.font.name = 'Times New Roman'
        run.font.size = base_font_size
        run.bold = default_bold or t.get('bold', False)
        run.italic = default_italic or t.get('italic', False)
        if t.get('code', False):
            run.font.name = 'Consolas'
            run.font.size = Pt(base_font_size.pt * 0.85)
            run.font.color.rgb = COLOR_CODE
        else:
            run.font.color.rgb = base_color

def set_cell_properties(cell, fill_hex=None, top_dxa=25, bottom_dxa=25, left_dxa=50, right_dxa=50, border_color="D0D5DD"):
    tcPr = cell._tc.get_or_add_tcPr()
    if border_color:
        bdr_xml = parse_xml(f'''
            <w:tcBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>
                <w:left w:val="none"/>
                <w:bottom w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>
                <w:right w:val="none"/>
            </w:tcBorders>
        ''')
        tcPr.append(bdr_xml)
        
    if fill_hex:
        shd_xml = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd_xml)
        
    mar_xml = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top_dxa}" w:type="dxa"/>
            <w:bottom w:w="{bottom_dxa}" w:type="dxa"/>
            <w:left w:w="{left_dxa}" w:type="dxa"/>
            <w:right w:w="{right_dxa}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(mar_xml)

def setup_document_styles(doc):
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(10)
    style_normal.font.color.rgb = COLOR_TEXT

    for section in doc.sections:
        section.top_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
        
        sectPr = section._sectPr
        pgBdr_xml = parse_xml(f'''
            <w:pgBorders {nsdecls("w")} w:offsetFrom="page">
                <w:top w:val="single" w:sz="6" w:space="16" w:color="000000"/>
                <w:left w:val="single" w:sz="6" w:space="16" w:color="000000"/>
                <w:bottom w:val="single" w:sz="6" w:space="16" w:color="000000"/>
                <w:right w:val="single" w:sz="6" w:space="16" w:color="000000"/>
            </w:pgBorders>
        ''')
        sectPr.append(pgBdr_xml)
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        fr = fp.add_run("Higher Education Students' Performance Evaluation | Prajith Sabarish D (24AD079) | Page ")
        fr.font.name = "Times New Roman"
        fr.font.size = Pt(9)
        fr.font.italic = True
        fr.font.color.rgb = COLOR_TEXT
        
        fld = parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"><w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="18"/><w:i/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>')
        fp._p.append(fld)

def add_title_page(doc):
    p_top = doc.add_paragraph()
    p_top.paragraph_format.space_before = Pt(24)
    p_top.paragraph_format.space_after = Pt(6)

    p_t1 = doc.add_paragraph()
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_t1.add_run("HIGHER EDUCATION STUDENTS' PERFORMANCE EVALUATION:")
    r_t1.font.name = "Times New Roman"
    r_t1.font.size = Pt(15.5)
    r_t1.font.bold = True
    r_t1.font.color.rgb = COLOR_PRIMARY
    p_t1.paragraph_format.space_after = Pt(4)

    p_t2 = doc.add_paragraph()
    p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t2 = p_t2.add_run("A COMPREHENSIVE CASE STUDY IN EXPLORATORY DATA ANALYSIS, STATISTICAL INFERENCE, AND DEEP LEARNING ARCHITECTURES")
    r_t2.font.name = "Times New Roman"
    r_t2.font.size = Pt(13)
    r_t2.font.bold = True
    r_t2.font.color.rgb = COLOR_PRIMARY
    p_t2.paragraph_format.space_after = Pt(20)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_s1 = p_sub.add_run("A Case Study Technical Report submitted in fulfillment of the course assessment for\n")
    r_s1.font.name = "Times New Roman"
    r_s1.font.size = Pt(10.5)
    r_s1.font.italic = True
    r_s1.font.color.rgb = COLOR_PRIMARY

    r_s2 = p_sub.add_run("U21ADPO5 - EXPLORATORY DATA ANALYSIS AND VISUALIZATION")
    r_s2.font.name = "Times New Roman"
    r_s2.font.size = Pt(11)
    r_s2.font.bold = True
    r_s2.font.color.rgb = COLOR_PRIMARY
    p_sub.paragraph_format.space_after = Pt(24)

    logo_path = os.path.join("images", "kpr_logo.png")
    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_after = Pt(4)
        run_logo = p_logo.add_run()
        run_logo.add_picture(logo_path, width=Inches(2.2))

    p_addr = doc.add_paragraph()
    p_addr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_addr = p_addr.add_run("Avinashi Road, Arasur, Coimbatore.")
    r_addr.font.name = "Times New Roman"
    r_addr.font.size = Pt(10)
    r_addr.font.color.rgb = COLOR_TEXT
    p_addr.paragraph_format.space_after = Pt(28)

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(16)
    p_meta.paragraph_format.space_after = Pt(2)
    p_meta.paragraph_format.line_spacing = 1.15
    
    r_c1 = p_meta.add_run("Candidate: ")
    r_c1.font.name = "Times New Roman"
    r_c1.font.bold = True
    r_c1.font.size = Pt(9.5)
    
    r_c2 = p_meta.add_run("Prajith Sabarish D (24AD079)  |  Department of Artificial Intelligence & Data Science\n")
    r_c2.font.name = "Times New Roman"
    r_c2.font.size = Pt(9.5)

    r_gh1 = p_meta.add_run("GitHub Repository: ")
    r_gh1.font.name = "Times New Roman"
    r_gh1.font.bold = True
    r_gh1.font.size = Pt(9.5)
    
    r_gh2 = p_meta.add_run("https://github.com/PrajithSabarish/Higher-Education-Students-Performance-Evaluation\n")
    r_gh2.font.name = "Times New Roman"
    r_gh2.font.size = Pt(9.5)

    r_ds1 = p_meta.add_run("Dataset Benchmark URL: ")
    r_ds1.font.name = "Times New Roman"
    r_ds1.font.bold = True
    r_ds1.font.size = Pt(9.5)

    r_ds2 = p_meta.add_run("https://archive.ics.uci.edu/dataset/582/higher+education+students+performance+evaluation")
    r_ds2.font.name = "Times New Roman"
    r_ds2.font.size = Pt(9.5)

    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.paragraph_format.space_before = Pt(28)
    r_date = p_date.add_run("OCTOBER 2026")
    r_date.font.name = "Times New Roman"
    r_date.font.size = Pt(10.5)
    r_date.font.bold = True
    r_date.font.color.rgb = COLOR_TEXT

    doc.add_page_break()

def add_abstract_page(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_h.paragraph_format.space_before = Pt(14)
    p_h.paragraph_format.space_after = Pt(8)
    r_h = p_h.add_run("ABSTRACT")
    r_h.font.name = "Times New Roman"
    r_h.font.size = Pt(12)
    r_h.font.bold = True
    r_h.font.color.rgb = COLOR_PRIMARY

    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_after = Pt(8)
    p_abs.paragraph_format.line_spacing = 1.15
    
    abs_text = (
        "Accurate early prediction of student academic attainment is essential for timely pedagogical intervention, "
        "personalized learning support, and institutional retention in higher education analytics. This case study "
        "investigates an end-to-end predictive modeling and visualization pipeline on the UCI Higher Education Students' "
        "Performance Evaluation benchmark dataset (145 students across 33 demographic, socioeconomic, and behavioral attributes). "
        "Through parametric and discrete distribution modeling (Binomial cohort analysis and Gaussian normality diagnostics), "
        "non-parametric association testing (Spearman rank correlation and Cramer's V), and Ben Fry's seven stages of data visualization, "
        "the behavioral feature space is systematically explored and synthesized into four domain-engineered composite indices: "
        "Study Engagement, Class Behaviour, Family Education, and Aspiration Gap. Benchmarking five modeling paradigms demonstrates that "
        "a Regularized Deep Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout achieves a 5-fold cross-validated accuracy "
        "of 68.3% ± 5.2% and a Macro-F1 score of 0.654 ± 0.058 on the 3-class academic risk classification task (Low, Medium, High), "
        "outperforming baseline Logistic Regression (64.8%) and Random Forest (66.2%). Dual-perspective feature importance demonstrates "
        "that student cumulative grade history (Prior GPA) and weekly study duration represent the decisive predictive drivers of academic success. "
        "All visual, tabular, and model artifacts are verified under strict zero-leakage standards."
    )
    r_a = p_abs.add_run(abs_text)
    r_a.font.name = "Times New Roman"
    r_a.font.size = Pt(10)

    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_before = Pt(4)
    p_kw.paragraph_format.space_after = Pt(8)
    r_kw_lbl = p_kw.add_run("Keywords: ")
    r_kw_lbl.font.name = "Times New Roman"
    r_kw_lbl.font.size = Pt(10)
    r_kw_lbl.font.bold = True
    
    r_kw_val = p_kw.add_run("Exploratory Data Analysis, Higher Education Analytics, Student Performance Prediction, Tabular Deep Learning, Multi-Layer Perceptron, Ben Fry Visualization Stages, Ordinal Evaluation.")
    r_kw_val.font.name = "Times New Roman"
    r_kw_val.font.size = Pt(10)

    doc.add_page_break()

def add_toc_pages(doc):
    p_h = doc.add_paragraph()
    p_h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_h.paragraph_format.space_before = Pt(8)
    p_h.paragraph_format.space_after = Pt(4)
    r_h = p_h.add_run("TABLE OF CONTENTS & ROADMAP")
    r_h.font.name = "Times New Roman"
    r_h.font.size = Pt(12)
    r_h.font.bold = True
    r_h.font.color.rgb = COLOR_PRIMARY

    toc_entries = [
        ("1. INTRODUCTION & OBJECTIVES", "4", True),
        ("  1.1 Context and Operational Motivation", "4", False),
        ("  1.2 Problem Statement & Mathematical Formulation", "4", False),
        ("  1.3 Project Objectives (Targeting CO3, CO4, CO5)", "4", False),
        ("  1.4 Summary of Technical Contributions", "5", False),
        ("  1.5 Report Organization & Roadmap", "5", False),
        ("  1.6 Literature Review (5 Sources)", "5", False),
        ("2. DATASET DESCRIPTION & FEATURE ENGINEERING", "6", True),
        ("  2.1 Source and Collection Context", "6", False),
        ("  2.2 Dataset Dimensions & Memory Footprint", "6", False),
        ("  2.3 Feature Dictionary (Table 3)", "7", False),
        ("  2.4 Summary Statistics for Academic Cohort (Table 5)", "7", False),
        ("  2.5 Target Variable Distribution (GRADE)", "8", False),
        ("  2.6 Grade_Group Target Tier Definition (Table 4)", "8", False),
        ("  2.7 Complete Engineered Feature Space (Table 7)", "9", False),
        ("3. EXPLORATORY DATA ANALYSIS & PREPROCESSING DECISIONS", "9", True),
        ("  3.1 Missing Value Mechanism & Imputation Protocols", "9", False),
        ("  3.2 Outlier Treatment Justification & Operational Sanity", "10", False),
        ("  3.3 Deduplication & Physical Sanity Validations", "10", False),
        ("  3.4 Skewness Rectification & Normalizing Transforms", "11", False),
        ("  3.5 Categorical Encodings & Leakage Avoidance Protocols", "11", False),
        ("  3.6 Stratified Data Partitioning & Generalisation Guarantees", "12", False),
        ("  3.7 Statistical Distribution Modeling & Goodness-of-Fit (Table 8 & 9)", "12", False),
        ("  3.8 Hypothesis Testing & Effect Size Quantifications (Table 10)", "13", False),
        ("4. COMPREHENSIVE DATA VISUALIZATION CATALOG (CO3 & CO4)", "13", True),
        ("  4.1 Ben Fry's Seven Stages of Data Visualization (Table 1)", "13", False),
        ("  4.2 Suitability Rationale for Selected Graphical Typologies (Table 2)", "14", False),
        ("  4.3 Exploratory Figure Catalog (Figures 1 to 22 & Fig. R)", "14", False),
        ("  4.4 Interactive Visualization Architecture (Table 19)", "24", False),
        ("  4.5 Bivariate and Multivariate Analytical Synthesis", "24", False),
        ("5. DEEP LEARNING MODEL & BASELINE ARCHITECTURES", "25", True),
        ("  5.1 Deep Learning Multi-Layer Perceptron (MLP) Architecture", "25", False),
        ("  5.2 Layer-by-Layer Architectural Specifications (Table 12)", "25", False),
        ("  5.3 Neural Training Regimen & Optimization Hyperparameters", "26", False),
        ("  5.4 Hyperparameter Search Space and Tuning Trials (Table 14)", "26", False),
        ("  5.5 Comparative Baseline Paradigms", "27", False),
        ("  5.6 Model Selection Justification: MLP vs Tree Ensembles", "27", False),
        ("6. RESULT VISUALIZATION & PERFORMANCE INTERPRETATION (CO5)", "28", True),
        ("  6.1 Multi-Metric Performance Benchmark (Table 15)", "28", False),
        ("  6.2 Rigorous Diagnostic Visualizations (Figures 23 to 33)", "29", False),
        ("  6.3 Performance Diagnosis: Feature Attribution, Error Partitioning & Ablation", "34", False),
        ("  6.4 Honest Discussion of Overfitting, Generalisation, and Sample Limitations", "34", False),
        ("7. CONCLUSION, PRACTICAL RECOMMENDATIONS & FUTURE SCOPE", "35", True),
        ("  7.1 Synthesis of Empirical Learnings", "35", False),
        ("  7.2 Practical Early-Warning and Academic Advising Framework (Table 18)", "35", False),
        ("  7.3 Operational Challenges and Technical Limitations", "36", False),
        ("  7.4 Concrete Future Research & Development Roadmap", "36", False),
        ("8. REFERENCES (IEEE FORMAT)", "37", True),
        ("9. APPENDIX: CODEBASE, REPOSITORY ARCHITECTURE & README", "37", True),
    ]

    tbl_toc = doc.add_table(rows=len(toc_entries), cols=2)
    tbl_toc.style = 'Table Grid'
    tbl_toc.alignment = WD_TABLE_ALIGNMENT.CENTER

    for idx, (title, page_num, is_major) in enumerate(toc_entries):
        c0 = tbl_toc.cell(idx, 0)
        c1 = tbl_toc.cell(idx, 1)
        c0.width = Inches(6.4)
        c1.width = Inches(0.7)

        bg_color = HEX_PRIMARY if is_major else None
        bdr_color = HEX_PRIMARY if is_major else "E2E8F0"
        set_cell_properties(c0, fill_hex=bg_color, top_dxa=18, bottom_dxa=18, left_dxa=50, right_dxa=50, border_color=bdr_color)
        set_cell_properties(c1, fill_hex=bg_color, top_dxa=18, bottom_dxa=18, left_dxa=50, right_dxa=50, border_color=bdr_color)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(title)
        r0.font.name = "Times New Roman"
        r0.font.size = Pt(8.5)
        r0.font.bold = is_major
        r0.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if is_major else COLOR_TEXT

        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(page_num)
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(8.5)
        r1.font.bold = is_major
        r1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if is_major else COLOR_TEXT

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(6)
    p_sp.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

def add_heading_styled(doc, level, text):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    
    if level == 1:
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        p.style = doc.styles['Heading 1']
        p.runs[0].text = text
    elif level == 2:
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(1.5)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
        p.style = doc.styles['Heading 2']
        p.runs[0].text = text
    elif level == 3:
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = COLOR_SECONDARY
        p.style = doc.styles['Heading 3']
        p.runs[0].text = text
    else:
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = COLOR_MUTED
    return p

def add_paragraph_styled(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1.8)
    p.paragraph_format.line_spacing = 1.06
    add_styled_runs(p, text, base_font_size=Pt(10), base_color=COLOR_TEXT)
    return p

def add_list_item_styled(doc, line):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.06

    indent_level = 0
    l_strip = line.lstrip()
    indent_spaces = len(line) - len(l_strip)
    if indent_spaces >= 4:
        indent_level = 2
    elif indent_spaces >= 2:
        indent_level = 1

    m_num = re.match(r'^(\d+[\.\)])\s+(.*)', l_strip)
    m_bullet = re.match(r'^([\-\*\+])\s+(.*)', l_strip)

    if m_num:
        prefix = m_num.group(1) + " "
        content = m_num.group(2)
    elif m_bullet:
        prefix = "• "
        content = m_bullet.group(2)
    else:
        prefix = "• "
        content = l_strip

    base_left = Inches(0.18 + indent_level * 0.18)
    p.paragraph_format.left_indent = base_left
    p.paragraph_format.first_line_indent = Inches(-0.12)

    r_pre = p.add_run(prefix)
    r_pre.font.name = "Times New Roman"
    r_pre.font.size = Pt(9.5)
    r_pre.font.bold = True
    r_pre.font.color.rgb = COLOR_PRIMARY

    add_styled_runs(p, content, base_font_size=Pt(9.5), base_color=COLOR_TEXT)
    return p

def add_table_styled(doc, table_lines):
    rows_data = []
    for l in table_lines:
        parts = [p.strip() for p in l.split('|')]
        if len(parts) >= 2 and parts[0] == '':
            parts = parts[1:]
        if len(parts) >= 1 and parts[-1] == '':
            parts = parts[:-1]
        rows_data.append(parts)

    if len(rows_data) < 2:
        return

    headers = rows_data[0]
    if len(rows_data) > 1 and all(set(c).issubset({'-', ':', ' '}) for c in rows_data[1]):
        data_rows = rows_data[2:]
    else:
        data_rows = rows_data[1:]

    num_cols = len(headers)
    num_rows = 1 + len(data_rows)

    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    total_width = 7.1
    if num_cols == 2:
        col_widths = [1.9, 5.2]
    elif num_cols == 3:
        if 'No.' in headers[0]:
            col_widths = [0.8, 5.5, 0.8]
        elif 'Step' in headers[0]:
            col_widths = [0.8, 2.5, 3.8]
        else:
            col_widths = [1.8, 2.6, 2.7]
    elif num_cols == 4:
        if 'Original Header' in headers[0]:
            col_widths = [1.6, 1.6, 1.0, 2.9]
        elif 'Grade Code' in headers[0]:
            col_widths = [1.3, 2.2, 1.6, 2.0]
        else:
            col_widths = [1.775] * 4
    elif num_cols == 5:
        col_widths = [1.42] * 5
    elif num_cols == 6:
        col_widths = [1.18] * 6
    elif num_cols == 8:
        col_widths = [1.8, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.8]
    else:
        col_widths = [total_width / num_cols] * num_cols

    if len(col_widths) < num_cols:
        col_widths.extend([total_width / num_cols] * (num_cols - len(col_widths)))

    trPr0 = table.rows[0]._tr.get_or_add_trPr()
    trPr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

    hdr_cells = table.rows[0].cells
    for c_idx, h_text in enumerate(headers):
        if c_idx < len(hdr_cells):
            cell = hdr_cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            set_cell_properties(cell, fill_hex=HEX_PRIMARY, top_dxa=40, bottom_dxa=40, left_dxa=50, right_dxa=50, border_color="1F4E78")
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            if any(w in h_text for w in ['No.', 'Count', '%', 'Code', 'Page', 'Split', 'Rank', 'Macro']):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_styled_runs(p, h_text, base_font_size=Pt(7.5), base_color=RGBColor(0xFF, 0xFF, 0xFF), default_bold=True)

    for r_idx, row in enumerate(data_rows):
        row_obj = table.rows[r_idx + 1]
        trPr = row_obj._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        bg_color = HEX_ZEBRA if (r_idx % 2 == 1) else None

        row_cells = row_obj.cells
        for c_idx, val in enumerate(row):
            if c_idx < len(row_cells):
                cell = row_cells[c_idx]
                cell.width = Inches(col_widths[c_idx])
                set_cell_properties(cell, fill_hex=bg_color, top_dxa=25, bottom_dxa=25, left_dxa=45, right_dxa=45, border_color=HEX_BORDER)
                
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0

                if num_cols > 3 and c_idx >= 2 and any(char.isdigit() for char in val):
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif c_idx == 0 and num_cols <= 4 and len(val) < 10:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

                font_sz = Pt(6.5) if num_cols >= 8 else Pt(7)
                add_styled_runs(p, val, base_font_size=font_sz, base_color=COLOR_TEXT)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(2)

def add_code_block_styled(doc, code_lines, lang="python"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(7.1)
    set_cell_properties(cell, fill_hex=HEX_CODE_BG, top_dxa=40, bottom_dxa=40, left_dxa=70, right_dxa=70, border_color="CBD5E1")

    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(0)
    p0.paragraph_format.line_spacing = 1.0

    full_code = "\n".join(code_lines)
    r0 = p0.add_run(full_code)
    r0.font.name = "Consolas"
    r0.font.size = Pt(6.8)
    r0.font.color.rgb = COLOR_CODE

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(2)

def add_image_styled(doc, marker, fig_map):
    m = re.match(r'^\*\[Insert\s+(Fig\.\s*[^:]+):\s*([^\]]+)\]\*', marker)
    if not m:
        return

    fig_key = m.group(1).strip()
    img_rel_path = m.group(2).strip()

    img_path = os.path.normpath(img_rel_path)
    if not os.path.exists(img_path):
        print(f"Warning: Image file not found: {img_path}")
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(2.5)
    p_img.paragraph_format.space_after = Pt(1)
    p_img.paragraph_format.keep_with_next = True

    try:
        run_img = p_img.add_run()
        # Scale to 3.85 inches to guarantee 2 figures fit per page cleanly
        run_img.add_picture(img_path, width=Inches(3.85))
    except Exception as e:
        print(f"Error adding picture {img_path}: {e}")
        return

    caption_text = fig_map.get(fig_key, "")
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0.5)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True

    num_str = fig_key.replace('Fig.', '').strip()
    r_cap = p_cap.add_run(f"Fig. {num_str}. {caption_text}")
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(8.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = COLOR_TEXT

def parse_and_convert(md_path, docx_path):
    print(f"Reading markdown from {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    fig_map = {}
    table_match = re.search(r'## LIST OF FIGURES\s*\n\s*\| No\. \| Title \| Page \|\s*\n\|[-:\s|]+\|\s*\n((?:\|[^\n]+\|\s*\n)+)', md_text)
    if table_match:
        rows = table_match.group(1).strip().split('\n')
        for r in rows:
            parts = [p.strip() for p in r.split('|') if p.strip()]
            if len(parts) >= 2:
                fig_map[parts[0]] = parts[1]

    print(f"Extracted {len(fig_map)} figure captions.")

    doc = docx.Document()
    setup_document_styles(doc)

    lines = md_text.splitlines()
    total_lines = len(lines)

    # 1. Title Page (Page 1)
    add_title_page(doc)

    # 2. Abstract Page (Page 2)
    add_abstract_page(doc)

    # 3. Table of Contents & Roadmap (Pages 3-4)
    add_toc_pages(doc)

    start_idx = 0
    for idx, l in enumerate(lines):
        if l.strip() == '## 2. INTRODUCTION AND OBJECTIVE':
            start_idx = idx
            break

    print(f"Beginning main content compilation at line {start_idx+1}...")

    i = start_idx
    section_title_map = {
        "## 2. INTRODUCTION AND OBJECTIVE": "1. INTRODUCTION & OBJECTIVES",
        "## 3. DATASET DESCRIPTION": "2. DATASET DESCRIPTION & FEATURE ENGINEERING",
        "## 4. EXPLORATORY DATA ANALYSIS AND PREPROCESSING": "3. EXPLORATORY DATA ANALYSIS & PREPROCESSING DECISIONS",
        "## 5. DATA VISUALISATION": "4. COMPREHENSIVE DATA VISUALIZATION CATALOG (CO3 & CO4)",
        "## 6. DEEP LEARNING MODEL": "5. DEEP LEARNING MODEL & BASELINE ARCHITECTURES",
        "## 7. RESULT VISUALISATION AND INTERPRETATION": "6. RESULT VISUALIZATION & PERFORMANCE INTERPRETATION (CO5)",
        "## 8. CONCLUSION AND FUTURE SCOPE": "7. CONCLUSION, PRACTICAL RECOMMENDATIONS & FUTURE SCOPE",
        "## 9. REFERENCES": "8. REFERENCES (IEEE FORMAT)",
        "## 10. APPENDIX": "9. APPENDIX: CODEBASE, REPOSITORY ARCHITECTURE & README",
    }

    last_was_page_break = True

    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith('*End of Part') or stripped.startswith('*Submitted in partial fulfilment'):
            i += 1
            continue
        if stripped.startswith("# Higher Education Students' Performance Evaluation"):
            i += 1
            continue

        # Code block
        if stripped.startswith('```'):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < total_lines and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i].rstrip('\r\n'))
                i += 1
            if i < total_lines:
                i += 1
            add_code_block_styled(doc, code_lines, lang=lang)
            last_was_page_break = False
            continue

        # Table
        if stripped.startswith('|'):
            tbl_lines = []
            while i < total_lines and lines[i].strip().startswith('|'):
                tbl_lines.append(lines[i].strip())
                i += 1
            add_table_styled(doc, tbl_lines)
            last_was_page_break = False
            continue

        # Image marker
        if stripped.startswith('*[Insert') and stripped.endswith(']*'):
            add_image_styled(doc, stripped, fig_map)
            i += 1
            last_was_page_break = False
            continue

        # Explicit newpage - ignored to keep continuous academic flow
        if stripped == '\\newpage':
            i += 1
            continue

        # Horizontal rule
        if stripped in ['---', '***', '___']:
            i += 1
            continue

        # Headings
        if stripped.startswith('#'):
            h_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
            if h_match:
                raw_level = len(h_match.group(1))
                h_text = h_match.group(2).strip()

                full_h = stripped
                if full_h in section_title_map:
                    h_text = section_title_map[full_h]
                    level = 1
                elif raw_level == 2:
                    level = 1
                elif raw_level == 3:
                    level = 2
                elif raw_level == 4:
                    level = 3
                else:
                    level = 4

                add_heading_styled(doc, level, h_text)
                last_was_page_break = False
                i += 1
                continue

        # Lists
        m_list = re.match(r'^\s*(\d+[\.\)]|\-|\*|\+)\s+', line)
        if m_list:
            add_list_item_styled(doc, line)
            last_was_page_break = False
            i += 1
            continue

        # Regular paragraph
        p_lines = [stripped]
        i += 1
        while i < total_lines:
            next_l = lines[i]
            next_s = next_l.strip()
            if not next_s:
                break
            if next_s.startswith(('```', '|', '>', '*[Insert', '\\newpage', '---', '***', '___', '#')):
                break
            if re.match(r'^\s*(\d+[\.\)]|\-|\*|\+)\s+', next_l):
                break
            if next_s.startswith('*End of Part') or next_s.startswith("# Higher Education"):
                break
            p_lines.append(next_s)
            i += 1

        full_p_text = " ".join(p_lines)
        add_paragraph_styled(doc, full_p_text)
        last_was_page_break = False

    print(f"Saving compiled document to {docx_path}...")
    doc.save(docx_path)
    print("Done! Document successfully generated.")

if __name__ == '__main__':
    md_file = os.path.join('reports', 'complete_project_report.md')
    docx_file = os.path.join('reports', 'final_report.docx')
    parse_and_convert(md_file, docx_file)
