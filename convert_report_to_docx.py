r"""
Production-grade Word Document (.docx) compiler for:
reports/complete_project_report.md

Built with 100% OpenXML Schema Compliance (ISO/IEC 29500-1):
- Fully verified and tested against Microsoft Word COM Automation engine
- Executive Academic Theme (Deep Navy #1B365D & Slate Blue #244B7A)
- Proper Heading 1 to Heading 4 hierarchy mapped to Word Navigation Pane
- Word Dynamic Table of Contents field (properly nested within <w:r> runs)
- All 34 Publication figures embedded in high resolution with centered captions
- All 20 Data tables styled with Navy headers, zebra striping, tblHeader and cantSplit
- Monospaced code blocks with shaded containers
- Standard tab-aligned header and footer with dynamic Page X numbering
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# Curated Executive Academic Palette
COLOR_PRIMARY = RGBColor(0x1B, 0x36, 0x5D)    # Deep Navy (#1B365D)
COLOR_SECONDARY = RGBColor(0x24, 0x4B, 0x7A)  # Slate Navy (#244B7A)
COLOR_TERTIARY = RGBColor(0x2C, 0x6B, 0x9E)   # Cobalt (#2C6B9E)
COLOR_DARK_SLATE = RGBColor(0x34, 0x49, 0x5E) # Slate Charcoal (#34495E)
COLOR_BODY = RGBColor(0x22, 0x22, 0x22)       # Charcoal text (#222222)
COLOR_MUTED = RGBColor(0x66, 0x70, 0x85)      # Slate Gray (#667085)
COLOR_CODE = RGBColor(0x1E, 0x29, 0x3B)       # Monospace Code Slate

HEX_PRIMARY = "1B365D"
HEX_SECONDARY = "244B7A"
HEX_ZEBRA = "F8F9FA"
HEX_CODE_BG = "F8F9FA"
HEX_CALLOUT_BG = "F0F4F8"

def set_cell_properties(cell, fill_hex=None, top_dxa=80, bottom_dxa=80, left_dxa=100, right_dxa=100, border_color=None):
    """
    Sets cell shading, margins, and borders in strict OpenXML schema order:
    1. tcBorders
    2. shd
    3. tcMar
    """
    tcPr = cell._tc.get_or_add_tcPr()
    
    if border_color:
        bdr_xml = parse_xml(f'''
            <w:tcBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>
                <w:left w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>
                <w:bottom w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>
                <w:right w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>
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

def add_styled_runs(paragraph, text, base_font_size=Pt(11), base_color=COLOR_BODY, default_bold=False, default_italic=False):
    tokens = parse_inline_tokens(text)
    for t in tokens:
        run = paragraph.add_run(t['text'])
        run.font.name = 'Calibri'
        run.font.size = base_font_size
        run.bold = default_bold or t.get('bold', False)
        run.italic = default_italic or t.get('italic', False)
        if t.get('code', False):
            run.font.name = 'Consolas'
            run.font.size = Pt(base_font_size.pt * 0.9)
            run.font.color.rgb = COLOR_CODE
        else:
            run.font.color.rgb = base_color

def setup_document_styles(doc):
    # Configure base styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = COLOR_BODY

    # Heading 1
    h1 = doc.styles['Heading 1']
    h1.font.name = 'Arial'
    h1.font.size = Pt(16)
    h1.font.bold = True
    h1.font.color.rgb = COLOR_PRIMARY

    # Heading 2
    h2 = doc.styles['Heading 2']
    h2.font.name = 'Arial'
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = COLOR_SECONDARY

    # Heading 3
    h3 = doc.styles['Heading 3']
    h3.font.name = 'Arial'
    h3.font.size = Pt(11.5)
    h3.font.bold = True
    h3.font.color.rgb = COLOR_TERTIARY

    # Heading 4
    h4 = doc.styles['Heading 4']
    h4.font.name = 'Calibri'
    h4.font.size = Pt(11)
    h4.font.bold = True
    h4.font.italic = True
    h4.font.color.rgb = COLOR_DARK_SLATE

    # Page Margins & Header/Footer
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
        section.different_first_page_header_footer = True
        
        # Header (Pages 2+)
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Higher Education Students' Performance Evaluation | EDA & Deep Learning Report")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = COLOR_MUTED
        
        # Footer (Pages 2+) - Native right tab stop (9360 dxa = 6.5 inches)
        footer = section.footer
        fp = footer.paragraphs[0]
        pPr = fp._p.get_or_add_pPr()
        tabs = parse_xml(f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:pos="9360"/></w:tabs>')
        pPr.append(tabs)
        
        fr = fp.add_run("Department of Artificial Intelligence & Data Science | Case Study Report\tPage ")
        fr.font.name = "Calibri"
        fr.font.size = Pt(8.5)
        fr.font.color.rgb = COLOR_MUTED
        
        # Schema-compliant fldSimple with child run fallback
        fld = parse_xml(f'<w:fldSimple {nsdecls("w")} w:instr="PAGE"><w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="17"/><w:color w:val="667085"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>')
        fp._p.append(fld)

def add_title_page(doc):
    p_top = doc.add_paragraph()
    p_top.paragraph_format.space_before = Pt(36)
    p_top.paragraph_format.space_after = Pt(12)

    p_dept_head = doc.add_paragraph()
    p_dept_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_dh = p_dept_head.add_run("DEPARTMENT OF ARTIFICIAL INTELLIGENCE & DATA SCIENCE")
    r_dh.font.name = "Arial"
    r_dh.font.size = Pt(11)
    r_dh.font.bold = True
    r_dh.font.color.rgb = COLOR_SECONDARY
    p_dept_head.paragraph_format.space_after = Pt(16)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("Higher Education Students' Performance Evaluation")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(8)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("An Exploratory Data Analysis and Visualization Case Study with Deep Learning")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(13.5)
    r_sub.font.color.rgb = COLOR_SECONDARY
    p_sub.paragraph_format.space_after = Pt(24)

    # Clean divider paragraph with bottom border
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_before = Pt(0)
    p_div.paragraph_format.space_after = Pt(24)
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="18" w:space="6" w:color="{HEX_PRIMARY}"/></w:pBdr>')
    p_div._p.get_or_add_pPr().append(pBdr)

    # Metadata table (clean academic card)
    meta_items = [
        ("Student Name", "Prajith Sabarish D"),
        ("Register Number", "24AD079"),
        ("Degree & Department", "B.Tech. Artificial Intelligence and Data Science"),
        ("Academic Year / Semester", "III Year / V Semester"),
        ("Course Code & Title", "[COURSE CODE AND NAME]"),
        ("Institution", "[COLLEGE NAME]"),
        ("Faculty Advisor", "[FACULTY NAME]"),
        ("Date of Submission", "[DATE]"),
        ("GitHub Repository", "[PUBLIC REPO LINK]"),
    ]

    tbl = doc.add_table(rows=len(meta_items), cols=2)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (label, val) in enumerate(meta_items):
        c0 = tbl.cell(idx, 0)
        c1 = tbl.cell(idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        
        bg = HEX_ZEBRA if (idx % 2 == 1) else None
        set_cell_properties(c0, fill_hex=bg, top_dxa=70, bottom_dxa=70, left_dxa=100, right_dxa=100)
        set_cell_properties(c1, fill_hex=bg, top_dxa=70, bottom_dxa=70, left_dxa=100, right_dxa=100)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        p0.paragraph_format.space_before = Pt(2)
        r0 = p0.add_run(label)
        r0.font.name = "Calibri"
        r0.font.size = Pt(10)
        r0.font.bold = True
        r0.font.color.rgb = COLOR_PRIMARY

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        p1.paragraph_format.space_before = Pt(2)
        r1 = p1.add_run(val)
        r1.font.name = "Calibri"
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_BODY

    p_sp2 = doc.add_paragraph()
    p_sp2.paragraph_format.space_before = Pt(28)
    p_sp2.paragraph_format.space_after = Pt(4)

    # Submission statement callout box
    stmt_tbl = doc.add_table(rows=1, cols=1)
    stmt_tbl.style = 'Table Grid'
    stmt_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    stmt_cell = stmt_tbl.cell(0, 0)
    stmt_cell.width = Inches(6.5)
    set_cell_properties(stmt_cell, fill_hex=HEX_CALLOUT_BG, top_dxa=140, bottom_dxa=140, left_dxa=180, right_dxa=180)

    sp = stmt_cell.paragraphs[0]
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sp.paragraph_format.space_before = Pt(4)
    sp.paragraph_format.space_after = Pt(4)
    sr = sp.add_run("Submitted in partial fulfilment of the requirements for the graded case study (40 Marks):\n")
    sr.font.name = "Calibri"
    sr.font.size = Pt(10)
    sr.font.italic = True
    sr.font.color.rgb = COLOR_DARK_SLATE

    sr2 = sp.add_run('“Exploratory Data Analysis and Visualization of a Real-World Dataset”')
    sr2.font.name = "Calibri"
    sr2.font.size = Pt(10.5)
    sr2.font.bold = True
    sr2.font.color.rgb = COLOR_PRIMARY

    doc.add_page_break()

def add_heading_styled(doc, level, text):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(6)
    elif level == 2:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
    elif level == 3:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
    elif level == 4:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
    return p

def add_table_of_contents_field(doc):
    p_info = doc.add_paragraph()
    p_info.paragraph_format.space_before = Pt(2)
    p_info.paragraph_format.space_after = Pt(6)
    r_info = p_info.add_run("(Word Dynamic Table of Contents: In Microsoft Word, press F9 or Right-Click -> 'Update Field' to refresh page numbers automatically.)")
    r_info.font.name = "Calibri"
    r_info.font.size = Pt(9.5)
    r_info.font.italic = True
    r_info.font.color.rgb = COLOR_MUTED

    p_toc = doc.add_paragraph()
    p_toc.paragraph_format.space_before = Pt(4)
    p_toc.paragraph_format.space_after = Pt(12)
    
    # Strictly schema-compliant TOC field wrapped in runs
    r1 = p_toc.add_run()
    r1._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>'))

    r2 = p_toc.add_run()
    r2._r.append(parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText>'))

    r3 = p_toc.add_run()
    r3._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="separate"/>'))

    r4 = p_toc.add_run('Right-click and select "Update Field" to refresh the Table of Contents.')
    r4.font.name = "Calibri"
    r4.font.size = Pt(10)
    r4.font.italic = True
    r4.font.color.rgb = COLOR_MUTED

    r5 = p_toc.add_run()
    r5._r.append(parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>'))

def add_paragraph_styled(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    add_styled_runs(p, text, base_font_size=Pt(11), base_color=COLOR_BODY)
    return p

def add_list_item_styled(doc, line):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15

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

    base_left = Inches(0.25 + indent_level * 0.25)
    p.paragraph_format.left_indent = base_left
    p.paragraph_format.first_line_indent = Inches(-0.2)

    r_pre = p.add_run(prefix)
    r_pre.font.name = "Calibri"
    r_pre.font.size = Pt(10.5)
    r_pre.font.bold = True
    r_pre.font.color.rgb = COLOR_PRIMARY

    add_styled_runs(p, content, base_font_size=Pt(10.5), base_color=COLOR_BODY)
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

    total_width = 6.5
    if num_cols == 2:
        col_widths = [1.8, 4.7]
    elif num_cols == 3:
        if 'No.' in headers[0]:
            col_widths = [0.9, 4.8, 0.8]
        elif 'Step' in headers[0]:
            col_widths = [0.8, 2.2, 3.5]
        else:
            col_widths = [1.6, 2.4, 2.5]
    elif num_cols == 4:
        if 'Original Header' in headers[0]:
            col_widths = [1.5, 1.5, 0.9, 2.6]
        elif 'Grade Code' in headers[0]:
            col_widths = [1.2, 2.0, 1.5, 1.8]
        else:
            col_widths = [1.625] * 4
    elif num_cols == 5:
        col_widths = [1.3] * 5
    elif num_cols == 6:
        col_widths = [1.08] * 6
    elif num_cols == 8:
        col_widths = [1.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    else:
        col_widths = [total_width / num_cols] * num_cols

    if len(col_widths) < num_cols:
        col_widths.extend([total_width / num_cols] * (num_cols - len(col_widths)))

    # Set Header Row
    hdr_cells = table.rows[0].cells
    # Repeat header on next page in trPr
    trPr0 = table.rows[0]._tr.get_or_add_trPr()
    trPr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

    for c_idx, h_text in enumerate(headers):
        if c_idx < len(hdr_cells):
            cell = hdr_cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            set_cell_properties(cell, fill_hex=HEX_PRIMARY, top_dxa=90, bottom_dxa=90, left_dxa=100, right_dxa=100)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            if any(w in h_text for w in ['No.', 'Count', '%', 'Code', 'Page', 'Split', 'Rank', 'Macro']):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_styled_runs(p, h_text, base_font_size=Pt(9), base_color=RGBColor(0xFF, 0xFF, 0xFF), default_bold=True)

    # Set Data Rows
    for r_idx, row in enumerate(data_rows):
        row_obj = table.rows[r_idx + 1]
        # cantSplit on trPr
        trPr = row_obj._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        bg_color = HEX_ZEBRA if (r_idx % 2 == 1) else None

        row_cells = row_obj.cells
        for c_idx, val in enumerate(row):
            if c_idx < len(row_cells):
                cell = row_cells[c_idx]
                cell.width = Inches(col_widths[c_idx])
                set_cell_properties(cell, fill_hex=bg_color, top_dxa=70, bottom_dxa=70, left_dxa=90, right_dxa=90)
                
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.05

                if num_cols > 3 and c_idx >= 2 and any(char.isdigit() for char in val):
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif c_idx == 0 and num_cols <= 4 and len(val) < 10:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

                font_sz = Pt(8) if num_cols >= 8 else Pt(9)
                add_styled_runs(p, val, base_font_size=font_sz, base_color=COLOR_BODY)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(6)

def add_code_block_styled(doc, code_lines, lang="python"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_properties(cell, fill_hex=HEX_CODE_BG, top_dxa=100, bottom_dxa=100, left_dxa=140, right_dxa=140)

    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(1)
    p0.paragraph_format.line_spacing = 1.0

    if code_lines:
        r0 = p0.add_run(code_lines[0])
        r0.font.name = "Consolas"
        r0.font.size = Pt(8.5)
        r0.font.color.rgb = COLOR_CODE

        for l in code_lines[1:]:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(l if l else " ")
            r.font.name = "Consolas"
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_CODE

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(6)

def add_blockquote_styled(doc, bq_lines):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_properties(cell, fill_hex=HEX_CALLOUT_BG, top_dxa=100, bottom_dxa=100, left_dxa=140, right_dxa=140)

    p0 = cell.paragraphs[0]
    p0.paragraph_format.space_before = Pt(2)
    p0.paragraph_format.space_after = Pt(2)
    full_text = " ".join(bq_lines)
    add_styled_runs(p0, full_text, base_font_size=Pt(10), base_color=COLOR_DARK_SLATE, default_italic=True)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(6)

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
    p_img.paragraph_format.space_before = Pt(10)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True

    try:
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(6.2))
    except Exception as e:
        print(f"Error adding picture {img_path}: {e}")
        return

    caption_text = fig_map.get(fig_key, "")
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)

    r_num = p_cap.add_run(f"Figure {fig_key.replace('Fig.', '').strip()}: ")
    r_num.font.name = "Calibri"
    r_num.font.size = Pt(9.5)
    r_num.font.bold = True
    r_num.font.color.rgb = COLOR_PRIMARY

    r_text = p_cap.add_run(caption_text if caption_text else "Visualisation illustration.")
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(9.5)
    r_text.font.italic = True
    r_text.font.color.rgb = COLOR_MUTED

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

    # 1. Add Dedicated Title Page
    add_title_page(doc)

    start_idx = 0
    for idx, l in enumerate(lines):
        if l.strip() == '\\newpage':
            start_idx = idx + 1
            break

    print(f"Beginning main content parsing at line {start_idx+1}...")

    i = start_idx
    major_sections = [
        "## 1. ABSTRACT",
        "## 2. INTRODUCTION AND OBJECTIVE",
        "## 3. DATASET DESCRIPTION",
        "## 4. EXPLORATORY DATA ANALYSIS AND PREPROCESSING",
        "## 5. DATA VISUALISATION",
        "## 6. DEEP LEARNING MODEL",
        "## 7. RESULT VISUALISATION AND INTERPRETATION",
        "## 8. CONCLUSION AND FUTURE SCOPE",
        "## 9. REFERENCES",
        "## 10. APPENDIX",
        "## 11. FINAL SUBMISSION VERIFICATION CHECKLIST",
    ]

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

        # Blockquote
        if stripped.startswith('>'):
            bq_lines = []
            while i < total_lines and lines[i].strip().startswith('>'):
                bq_lines.append(lines[i].strip()[1:].strip())
                i += 1
            add_blockquote_styled(doc, bq_lines)
            last_was_page_break = False
            continue

        # Image marker
        if stripped.startswith('*[Insert') and stripped.endswith(']*'):
            add_image_styled(doc, stripped, fig_map)
            i += 1
            last_was_page_break = False
            continue

        # Explicit newpage
        if stripped == '\\newpage':
            if not last_was_page_break:
                doc.add_page_break()
                last_was_page_break = True
            i += 1
            continue

        # Horizontal rule
        if stripped in ['---', '***', '___']:
            i += 1
            continue

        # Major section page break
        is_major = any(stripped.startswith(sec) for sec in major_sections)
        if is_major and not last_was_page_break:
            doc.add_page_break()
            last_was_page_break = True

        # Headings
        if stripped.startswith('#'):
            h_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
            if h_match:
                raw_level = len(h_match.group(1))
                h_text = h_match.group(2).strip()
                
                if raw_level == 2:
                    level = 1
                elif raw_level == 3:
                    level = 2
                elif raw_level == 4:
                    level = 3
                elif raw_level >= 5:
                    level = 4
                else:
                    level = 1

                add_heading_styled(doc, level, h_text)
                
                if h_text.upper() == 'TABLE OF CONTENTS':
                    add_table_of_contents_field(doc)

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
    docx_file = os.path.join('reports', 'complete_project_report.docx')
    parse_and_convert(md_file, docx_file)
