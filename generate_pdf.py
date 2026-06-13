import os
import subprocess
import time
import markdown

def main():
    md_file = "technical_whitepaper.md"
    html_file = "temp_whitepaper.html"
    pdf_file = "technical_whitepaper_v3.pdf"
    
    print(f"[1/4] Reading Markdown file: {md_file}")
    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    print("[2/4] Converting Markdown to HTML with Tables and Code Extensions...")
    # Convert markdown to html using standard extensions
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    # Beautiful CSS template for A4 page printing with Microsoft JhengHei
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>50家創業公司特徵選擇與機器學習預測之商業智慧白皮書</title>
    <!-- MathJax for rendering LaTeX formulas -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        @page {{
            size: A4;
            margin: 2.5cm 2cm 2.5cm 2cm;
            @bottom-center {{
                content: counter(page);
            }}
        }}
        
        body {{
            font-family: "Microsoft JhengHei", "微軟正黑體", sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333333;
            background-color: #ffffff;
        }}
        
        /* Cover Page Styling */
        .cover-page {{
            page-break-after: always;
            text-align: center;
            padding-top: 5cm;
        }}
        .cover-title {{
            font-size: 26pt;
            font-weight: bold;
            color: #1a365d;
            margin-bottom: 0.5cm;
            line-height: 1.3;
        }}
        .cover-subtitle {{
            font-size: 16pt;
            color: #4a5568;
            margin-bottom: 2cm;
        }}
        .cover-meta {{
            font-size: 12pt;
            color: #718096;
            margin-top: 5cm;
            line-height: 1.8;
        }}
        
        /* Headings */
        h1 {{
            font-size: 20pt;
            color: #1a365d;
            border-bottom: 2px solid #2b6cb0;
            padding-bottom: 0.3cm;
            margin-top: 1.5cm;
            margin-bottom: 0.8cm;
            page-break-before: always;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 15pt;
            color: #2b6cb0;
            border-bottom: 1px dashed #cbd5e0;
            padding-bottom: 0.2cm;
            margin-top: 1cm;
            margin-bottom: 0.6cm;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 12pt;
            color: #2d3748;
            margin-top: 0.8cm;
            margin-bottom: 0.4cm;
            page-break-after: avoid;
        }}
        h4 {{
            font-size: 11pt;
            color: #4a5568;
            margin-top: 0.6cm;
            margin-bottom: 0.3cm;
            page-break-after: avoid;
        }}
        
        p {{
            margin-bottom: 0.5cm;
            text-align: justify;
        }}
        
        /* Lists */
        ul, ol {{
            margin-bottom: 0.5cm;
            padding-left: 0.8cm;
            page-break-inside: avoid;
        }}
        li {{
            margin-bottom: 0.2cm;
        }}
        
        /* Code Blocks */
        pre {{
            background-color: #f7fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #4299e1;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            margin-bottom: 0.5cm;
            font-family: "Consolas", "Courier New", monospace;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        code {{
            font-family: "Consolas", "Courier New", monospace;
            font-size: 9.5pt;
            background-color: #edf2f7;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.6cm;
            margin-bottom: 0.8cm;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        tr {{
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #cbd5e0;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #ebf8ff;
            color: #2b6cb0;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f7fafc;
        }}
        
        /* Blockquotes / Alerts */
        blockquote {{
            background-color: #fffff0;
            border-left: 4px solid #d69e2e;
            padding: 10px 15px;
            margin: 0.5cm 0;
            font-style: italic;
            color: #744210;
            page-break-inside: avoid;
        }}
        
        /* Images styling */
        img {{
            max-width: 90%;
            height: auto;
            display: block;
            margin: 1cm auto;
            border: 1px solid #cbd5e0;
            padding: 6px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            page-break-inside: avoid;
        }}
        
        /* Centered caption for images */
        .image-caption {{
            text-align: center;
            font-size: 9.5pt;
            color: #718096;
            margin-top: -0.8cm;
            margin-bottom: 0.8cm;
            font-style: italic;
            page-break-before: avoid;
        }}
        
        hr {{
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 1.5cm 0;
        }}
        
        /* Class for hiding print headers */
        .no-print {{
            display: none;
        }}
    </style>
</head>
<body>

    <!-- Cover Page -->
    <div class="cover-page">
        <div class="cover-title">50家創業公司特徵選擇與機器學習預測之商業智慧白皮書</div>
        <div class="cover-subtitle">基於 CRISP-DM 流程、多重共線性分析與五大特徵篩選演算法的實證研究</div>
    </div>

    <!-- Main Content -->
    <div class="content-body">
        {html_body}
    </div>

    <script>
        // Make sure image captions are printed nicely
        document.querySelectorAll('img').forEach(img => {{
            const captionText = img.getAttribute('alt');
            if (captionText) {{
                const captionDiv = document.createElement('div');
                captionDiv.className = 'image-caption';
                captionDiv.innerText = captionText;
                img.parentNode.insertBefore(captionDiv, img.nextSibling);
            }}
        }});
    </script>
</body>
</html>
"""
    
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"[INFO] HTML file written to: {html_file}")
    
    print("[3/4] Running Headless Chrome to print to PDF...")
    chrome_path = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    
    # Run Google Chrome CLI command to print to PDF
    # --no-sandbox might be needed for certain permission scopes
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={os.path.abspath(pdf_file)}",
        os.path.abspath(html_file)
    ]
    
    # Wait 2 seconds to make sure MathJax has a bit of time to settle in headless rendering
    time.sleep(2)
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[4/4] PDF generated successfully: {pdf_file}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Chrome print-to-pdf failed: {e}")
        print(f"Stderr: {e.stderr.decode('utf-8', errors='ignore')}")
    finally:
        # Clean up temporary HTML file
        if os.path.exists(html_file):
            os.remove(html_file)
            print("[INFO] Cleaned up temporary HTML file.")

if __name__ == "__main__":
    main()
