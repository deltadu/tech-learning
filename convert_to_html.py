#!/usr/bin/env python3
"""
Convert all markdown files to print-friendly HTML.

Usage:
    pip install markdown pygments
    python convert_to_html.py

Output: Creates .html file next to each .md file
"""

import markdown
from pathlib import Path
import re

# HTML template with print-friendly CSS
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* Print-friendly styling */
        :root {{
            --text-color: #1a1a1a;
            --bg-color: #ffffff;
            --code-bg: #f5f5f5;
            --border-color: #ddd;
            --link-color: #0066cc;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: var(--text-color);
            background: var(--bg-color);
            max-width: 8.5in;
            margin: 0 auto;
            padding: 0.5in;
        }}
        
        h1 {{
            font-size: 24pt;
            border-bottom: 2px solid var(--text-color);
            padding-bottom: 0.3em;
            margin-top: 0;
        }}
        
        h2 {{
            font-size: 18pt;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.2em;
            margin-top: 1.5em;
            page-break-after: avoid;
        }}
        
        h3 {{
            font-size: 14pt;
            margin-top: 1.2em;
            page-break-after: avoid;
        }}
        
        h4 {{
            font-size: 12pt;
            margin-top: 1em;
        }}
        
        p {{
            margin: 0.8em 0;
        }}
        
        a {{
            color: var(--link-color);
            text-decoration: none;
        }}
        
        /* Code blocks */
        pre {{
            background: var(--code-bg);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 12px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }}
        
        code {{
            font-family: "SF Mono", "Consolas", "Monaco", monospace;
            font-size: 9pt;
        }}
        
        p code, li code, td code {{
            background: var(--code-bg);
            padding: 2px 5px;
            border-radius: 3px;
            border: 1px solid var(--border-color);
        }}
        
        pre code {{
            background: none;
            padding: 0;
            border: none;
        }}
        
        /* Tables */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            text-align: left;
        }}
        
        th {{
            background: var(--code-bg);
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{
            background: #fafafa;
        }}
        
        /* Lists */
        ul, ol {{
            padding-left: 1.5em;
            margin: 0.5em 0;
        }}
        
        li {{
            margin: 0.3em 0;
        }}
        
        /* Blockquotes */
        blockquote {{
            border-left: 4px solid var(--border-color);
            margin: 1em 0;
            padding: 0.5em 1em;
            background: #fafafa;
        }}
        
        /* Horizontal rule */
        hr {{
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 2em 0;
        }}
        
        /* ASCII diagrams - preserve formatting */
        pre {{
            white-space: pre;
            word-wrap: normal;
        }}
        
        /* Print styles */
        @media print {{
            body {{
                padding: 0;
                font-size: 10pt;
            }}
            
            pre {{
                font-size: 8pt;
                border: 1px solid #999;
            }}
            
            h1 {{
                font-size: 20pt;
            }}
            
            h2 {{
                font-size: 16pt;
                page-break-after: avoid;
            }}
            
            h3 {{
                font-size: 13pt;
                page-break-after: avoid;
            }}
            
            pre, table, blockquote {{
                page-break-inside: avoid;
            }}
            
            a {{
                color: var(--text-color);
            }}
            
            /* Avoid orphans */
            p, li {{
                orphans: 3;
                widows: 3;
            }}
        }}
        
        /* Table of contents */
        .toc {{
            background: var(--code-bg);
            padding: 1em;
            border-radius: 4px;
            margin: 1em 0;
        }}
        
        .toc ul {{
            list-style: none;
            padding-left: 1em;
        }}
        
        .toc > ul {{
            padding-left: 0;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>
"""


def convert_md_to_html(md_path: Path) -> Path:
    """Convert a markdown file to HTML."""
    
    # Read markdown content
    md_content = md_path.read_text(encoding='utf-8')
    
    # Extract title from first heading
    title_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
    title = title_match.group(1) if title_match else md_path.stem
    
    # Convert markdown to HTML
    md = markdown.Markdown(
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br',
        ],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'guess_lang': False,
            }
        }
    )
    
    html_content = md.convert(md_content)
    
    # Wrap in template
    full_html = HTML_TEMPLATE.format(
        title=title,
        content=html_content
    )
    
    # Write HTML file
    html_path = md_path.with_suffix('.html')
    html_path.write_text(full_html, encoding='utf-8')
    
    return html_path


def main():
    """Convert all markdown files to HTML."""
    
    # Find all markdown files
    root = Path(__file__).parent
    md_files = list(root.rglob('*.md'))
    
    # Exclude this script's directory if any
    md_files = [f for f in md_files if '.git' not in str(f)]
    
    print(f"Found {len(md_files)} markdown files\n")
    
    converted = []
    errors = []
    
    for md_file in sorted(md_files):
        try:
            html_file = convert_md_to_html(md_file)
            rel_path = html_file.relative_to(root)
            print(f"✓ {rel_path}")
            converted.append(html_file)
        except Exception as e:
            print(f"✗ {md_file}: {e}")
            errors.append((md_file, e))
    
    print(f"\n{'='*50}")
    print(f"Converted: {len(converted)} files")
    if errors:
        print(f"Errors: {len(errors)} files")
    
    print("\nHTML files created. Open in browser and print (Cmd+P / Ctrl+P)")


if __name__ == "__main__":
    main()
