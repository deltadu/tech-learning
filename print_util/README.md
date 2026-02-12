# Print Utilities

Scripts for converting markdown and code files to print-friendly HTML.

## Setup

```bash
pip install markdown pygments
```

## Scripts

### 1. Convert Markdown to HTML

Converts all `.md` files in the project to HTML with print-friendly styling.

```bash
cd print_util
python convert_to_html.py
```

**Output:** Creates `html/` subfolder in each topic directory.

### 2. Convert Python Code to HTML

Converts Python files to syntax-highlighted HTML, optimized for 4-up printing.

```bash
cd print_util
python print_code.py                  # All .py files in project
python print_code.py ../python/       # Specific folder
python print_code.py ../file.py       # Specific file
```

**Output:** Creates `output/` folder with HTML files.

## Printing Tips

### For Markdown (documentation)

1. Open HTML file in browser
2. **Cmd+P** (Mac) or **Ctrl+P** (Windows)
3. Print or Save as PDF

### For Code (4 pages per sheet)

1. Open HTML file in browser
2. **Cmd+P** (Mac) or **Ctrl+P** (Windows)
3. Click **"More settings"**
4. Set **"Pages per sheet"** → **4**
5. Print!

## Output Structure

```
print_util/
├── convert_to_html.py    # Markdown → HTML
├── print_code.py         # Python → HTML
├── README.md
└── output/               # Code HTML output
    ├── all_code.html     # Combined file
    └── python/           # Individual files
        └── ...
```

Topic folders get their own `html/` subfolder:
```
cryptography/
├── 01_FUNDAMENTALS.md
├── 02_PRACTICAL.md
└── html/
    ├── 01_FUNDAMENTALS.html
    └── 02_PRACTICAL.html
```
