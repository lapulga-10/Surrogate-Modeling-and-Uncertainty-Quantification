"""Convert a Jupyter Notebook into a standalone Medium-style HTML blog.

Usage:
    python notebook_to_medium.py path/to/notebook.ipynb [--output out.html] [--show-code]

This script uses nbformat and nbconvert.HTMLExporter to render the notebook to HTML,
then injects a Medium-like CSS and a simple hero header (title/lead paragraph) to make
output look more like a blog article suitable for publishing.
"""
import nbformat
from nbconvert import HTMLExporter
import argparse
import os
import re


def extract_title_and_lead(nb):
    """Extract a title and lead paragraph from the notebook's first markdown cell.
    Title: first line that starts with '# '
    Lead: first paragraph after the title or first paragraph in that cell.
    """
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            text = cell.source.strip()
            # search for a heading
            m = re.search(r"^#\s+(.+)", text, flags=re.MULTILINE)
            if m:
                title = m.group(1).strip()
                # remove title line and take the rest as lead
                rest = re.sub(r"^#\s+.+\n?", "", text, count=1).strip()
                # first paragraph
                lead = None
                for para in rest.split('\n\n'):
                    p = para.strip()
                    if p:
                        lead = p
                        break
                return title, lead
            else:
                # no header, take first non-empty paragraph as lead and use filename as title
                paras = [p.strip() for p in text.split('\n\n') if p.strip()]
                if paras:
                    return None, paras[0]
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('notebook', help='path to .ipynb file')
    p.add_argument('--output', '-o', help='output HTML file path', default=None)
    p.add_argument('--show-code', dest='show_code', action='store_true', help='keep code inputs visible')
    p.add_argument('--css', dest='css_path', default=os.path.join(os.path.dirname(__file__), '..', 'templates', 'medium.css'), help='path to CSS file')
    args = p.parse_args()

    nb_path = args.notebook
    if not os.path.exists(nb_path):
        raise FileNotFoundError(nb_path)

    out_path = args.output or os.path.splitext(nb_path)[0] + '_medium.html'

    print(f"Reading notebook: {nb_path}")
    nb = nbformat.read(nb_path, as_version=4)

    title, lead = extract_title_and_lead(nb)
    if not title:
        title = os.path.splitext(os.path.basename(nb_path))[0].replace('_', ' ')

    print(f"Detected title: {title}")

    # Load CSS
    css = ''
    if args.css_path and os.path.exists(args.css_path):
        with open(args.css_path, 'r', encoding='utf-8') as fh:
            css = fh.read()
            print(f"Loaded CSS from: {args.css_path}")
    else:
        print("Warning: CSS file not found, using minimal inline styles.")

    # Configure exporter
    exporter = HTMLExporter()
    # Optionally hide code inputs if not requested
    if not args.show_code:
        exporter.exclude_input = True

    print("Exporting notebook to HTML...")
    (body, resources) = exporter.from_notebook_node(nb)

    # Insert CSS into <head>
    if '</head>' in body:
        head_injection = f"<style>\n{css}\n</style>\n"
        body = body.replace('</head>', head_injection + '</head>', 1)
    else:
        # fallback: prepend style at top
        body = f"<style>\n{css}\n</style>\n" + body

    # Build hero header
    hero_html = f"""
<header class=\"header\">
  <h1 class=\"title\">{title}</h1>
  <div class=\"meta\">Converted from notebook · {os.path.basename(nb_path)}</div>
  {f'<p class=\"lead\">{lead}</p>' if lead else ''}
</header>
"""

    # Wrap the body content into our .container and insert hero after <body>
    if '<body' in body:
        # find <body ...>
        body = re.sub(r"<body([^>]*)>", r"<body\1>\n<div class=\"container\">" + hero_html + "<main class=\"article\">", body, count=1)
        # insert closing tags before </body>
        body = body.replace('</body>', '</main></div>\n</body>', 1)
    else:
        # fallback: just prepend container
        body = '<div class="container">' + hero_html + '<main class="article">' + body + '</main></div>'

    # Write output
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(body)

    print(f"Wrote Medium-style HTML to: {out_path}")


if __name__ == '__main__':
    main()
