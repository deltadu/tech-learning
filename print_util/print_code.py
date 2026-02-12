#!/usr/bin/env python3
"""
Convert Python files to print-friendly HTML optimized for 4-up printing.

Usage:
    pip install pygments
    cd print_util
    python print_code.py                     # All .py files in parent dir
    python print_code.py ../python/          # Specific folder
    python print_code.py ../file1.py         # Specific files

Output: Creates output/ folder with HTML files

Printing:
    1. Open the HTML file in browser
    2. Press Cmd+P (Mac) or Ctrl+P (Windows)
    3. Set "Pages per sheet" to 4
    4. Print!
"""

import sys
from pathlib import Path
from datetime import datetime

try:
    from pygments import highlight
    from pygments.lexers import get_lexer_for_filename, PythonLexer
    from pygments.formatters import HtmlFormatter
except ImportError:
    print("Please install pygments: pip install pygments")
    sys.exit(1)


# HTML template optimized for 4-up printing (larger font for readability)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        /* Base styles - optimized for 4-up printing */
        @page {{
            size: letter;
            margin: 0.3in;
            
            @bottom-center {{
                content: "{filename} - Page " counter(page);
                font-size: 8pt;
                color: #666;
            }}
        }}
        
        :root {{
            --font-size: 10pt;  /* Sized for 4-up printing */
            --line-height: 1.3;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: "Courier New", Courier, monospace;
            font-size: var(--font-size);
            letter-spacing: 0;
            font-variant-ligatures: none;
            line-height: var(--line-height);
            background: white;
            color: #1a1a1a;
        }}
        
        .file-section {{
            page-break-after: always;
            padding: 0.5em;
        }}
        
        .file-section:last-child {{
            page-break-after: avoid;
        }}
        
        .file-header {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 8px 12px;
            font-size: 10pt;
            font-weight: bold;
            margin-bottom: 0;
        }}
        
        .file-meta {{
            background: #3d3d3d;
            color: #999;
            padding: 4px 12px;
            font-size: 9pt;
        }}
        
        /* Syntax highlighting container */
        .highlight {{
            background: #fafafa;
            overflow-x: auto;
        }}
        
        .highlight pre {{
            margin: 0;
            padding: 12px;
            font-family: "Courier New", Courier, monospace;
            font-size: var(--font-size);
            line-height: var(--line-height);
            white-space: pre;
            word-wrap: normal;
            letter-spacing: 0;
            font-variant-ligatures: none;
        }}
        
        /* Line numbers - minimal */
        .linenodiv {{
            background: #f5f5f5;
            padding: 12px 3px 12px 2px;
            text-align: right;
            color: #aaa;
            font-size: 8pt;
            user-select: none;
            width: 1%;
            min-width: 1.5em;
            white-space: nowrap;
        }}
        
        .linenodiv pre {{
            margin: 0;
            line-height: var(--line-height);
        }}
        
        /* Code table layout */
        .highlighttable {{
            border-collapse: collapse;
            width: 100%;
        }}
        
        .highlighttable td {{
            vertical-align: top;
            padding: 0;
        }}
        
        .highlighttable td.code {{
            width: 100%;
        }}
        
        /* Footer on each page */
        .page-footer {{
            position: running(footer);
            font-size: 9pt;
            color: #666;
            text-align: center;
            padding-top: 0.5em;
        }}
        
        @media print {{
            body {{
                font-size: var(--font-size);
            }}
            
            .file-section {{
                page-break-after: always;
            }}
            
            .highlight {{
                background: white;
            }}
            
            /* Running footer */
            @page {{
                @bottom-center {{
                    content: "{filename} | Page " counter(page) " of " counter(pages);
                }}
            }}
        }}
        
        /* Pygments syntax highlighting - Light theme */
{pygments_css}
    </style>
</head>
<body>
{content}
</body>
</html>
"""

FILE_SECTION_TEMPLATE = """
<div class="file-section">
    <div class="file-header">{filename}</div>
    <div class="file-meta">{path} | {lines} lines | {modified}</div>
    {code}
</div>
"""


def get_pygments_css():
    """Get CSS for syntax highlighting."""
    formatter = HtmlFormatter(style='default', linenos=False)
    return formatter.get_style_defs('.highlight')


def highlight_code(code: str, filename: str) -> str:
    """Syntax highlight code and return HTML."""
    try:
        lexer = get_lexer_for_filename(filename)
    except:
        lexer = PythonLexer()
    
    formatter = HtmlFormatter(
        linenos='table',
        linenostart=1,
        cssclass='highlight',
        nowrap=False,
    )
    
    return highlight(code, lexer, formatter)


def convert_file(file_path: Path) -> str:
    """Convert a single file to HTML section."""
    code = file_path.read_text(encoding='utf-8', errors='replace')
    
    # Get file metadata
    stat = file_path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
    lines = len(code.splitlines())
    
    # Highlight code
    highlighted = highlight_code(code, file_path.name)
    
    return FILE_SECTION_TEMPLATE.format(
        filename=file_path.name,
        path=str(file_path),
        lines=lines,
        modified=modified,
        code=highlighted,
    )


def create_combined_html(files: list[Path], output_path: Path):
    """Create a single HTML file with all code files."""
    sections = []
    
    for file_path in sorted(files):
        try:
            section = convert_file(file_path)
            sections.append(section)
            print(f"  ✓ {file_path}")
        except Exception as e:
            print(f"  ✗ {file_path}: {e}")
    
    content = "\n".join(sections)
    
    html = HTML_TEMPLATE.format(
        title="Code Printout",
        filename="Multiple Files",
        pygments_css=get_pygments_css(),
        content=content,
    )
    
    output_path.write_text(html, encoding='utf-8')
    return output_path


def create_individual_html(file_path: Path, output_dir: Path) -> Path:
    """Create individual HTML file for a code file."""
    section = convert_file(file_path)
    
    html = HTML_TEMPLATE.format(
        title=file_path.name,
        filename=file_path.name,
        pygments_css=get_pygments_css(),
        content=section,
    )
    
    output_path = output_dir / f"{file_path.stem}.html"
    output_path.write_text(html, encoding='utf-8')
    return output_path


def find_code_files(
    paths: list[str], extensions: set = {".py"}
) -> list[Path]:
    """Find all code files in given paths."""
    files = []

    for path_str in paths:
        path = Path(path_str)

        if path.is_file() and path.suffix in extensions:
            files.append(path)
        elif path.is_dir():
            for ext in extensions:
                files.extend(path.rglob(f"*{ext}"))

    # Exclude common non-essential files
    exclude_patterns = [
        "__pycache__",
        ".git",
        "venv",
        "node_modules",
        ".egg",
    ]
    files = [
        f for f in files if not any(p in str(f) for p in exclude_patterns)
    ]

    return sorted(files)


def detect_extensions(paths: list[str]) -> set:
    """Detect which extensions to use based on paths."""
    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            return {path.suffix}
        elif path.is_dir():
            # Check directory name for hints
            name = path.name.lower()
            if name in ("cpp", "c++", "cxx"):
                return {".cpp", ".cc", ".h", ".hpp"}
            elif name in ("python", "py"):
                return {".py"}
            # Check what files exist in the directory
            cpp_files = list(path.rglob("*.cpp")) + list(path.rglob("*.cc"))
            py_files = list(path.rglob("*.py"))
            if cpp_files and not py_files:
                return {".cpp", ".cc", ".h", ".hpp"}
    return {".py"}  # default


def main():
    # Default: search parent directory (project root)
    if len(sys.argv) > 1:
        search_paths = sys.argv[1:]
    else:
        search_paths = [str(Path(__file__).parent.parent)]

    # Detect file extensions based on paths
    extensions = detect_extensions(search_paths)

    # Find code files
    files = find_code_files(search_paths, extensions)

    ext_str = ", ".join(extensions)
    if not files:
        print(f"No code files found ({ext_str}).")
        print("\nUsage:")
        print("  python print_code.py ../python/    # Python files")
        print("  python print_code.py ../cpp/       # C++ files")
        print("  python print_code.py ../file.cpp   # Specific file")
        return

    print(f"Found {len(files)} code files ({ext_str})\n")
    
    # Create output directory inside print_util
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create combined HTML
    print("Creating combined printout...")
    combined_path = create_combined_html(files, output_dir / "all_code.html")
    
    # Also create individual files
    print("\nCreating individual files...")
    
    # Find common base path for cleaner output structure
    project_root = Path(__file__).parent.parent
    
    for file_path in files:
        try:
            # Get path relative to project root, not cwd
            abs_path = file_path.resolve()
            try:
                rel_path = abs_path.relative_to(project_root)
            except ValueError:
                rel_path = Path(file_path.name)
            
            # Create subdirectory structure inside output
            sub_dir = output_dir / rel_path.parent
            sub_dir.mkdir(parents=True, exist_ok=True)
            
            html_path = sub_dir / f"{file_path.stem}.html"
            section = convert_file(file_path)
            html = HTML_TEMPLATE.format(
                title=file_path.name,
                filename=file_path.name,
                pygments_css=get_pygments_css(),
                content=section,
            )
            html_path.write_text(html, encoding='utf-8')
        except Exception as e:
            print(f"  ✗ {file_path}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Output: {output_dir.absolute()}")
    print(f"\nFiles created:")
    print(f"  • all_code.html (all files combined)")
    print(f"  • Individual HTML files in subfolders")
    print(f"\n{'='*50}")
    print("TO PRINT (4 pages per sheet):")
    print("  1. Open HTML file in browser")
    print("  2. Press Cmd+P (Mac) or Ctrl+P (Windows)")
    print("  3. Click 'More settings'")
    print("  4. Set 'Pages per sheet' to 4")
    print("  5. Print!")
    print("='*50")


if __name__ == "__main__":
    main()
