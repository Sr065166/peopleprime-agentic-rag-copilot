from pathlib import Path
import re
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import BASE_DIR, get_settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_no_hardcoded_secrets_in_frontend():
    """Verify that ADMIN_API_KEY, GROQ_API_KEY, PINECONE_API_KEY are NOT hardcoded in static files or HTML."""
    static_files = [
        BASE_DIR / "templates" / "index.html",
        BASE_DIR / "static" / "js" / "app.js",
        BASE_DIR / "static" / "css" / "style.css",
    ]

    forbidden_patterns = [
        re.compile(r"gsk_[a-zA-Z0-9]{20,}"),
        re.compile(r"pcsk_[a-zA-Z0-9]{20,}"),
        re.compile(r"tvly-[a-zA-Z0-9]{20,}"),
        re.compile(r'admin_api_key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
    ]

    for file_path in static_files:
        assert file_path.exists(), f"File {file_path} should exist"
        content = file_path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            match = pattern.search(content)
            assert match is None, f"Found forbidden secret pattern {pattern.pattern} in {file_path.name}"


@pytest.mark.anyio
async def test_frontend_serves_html_and_assets():
    """Verify that index.html, style.css, and app.js are correctly mounted and accessible via FastAPI."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Check index page
        res_html = await client.get("/")
        assert res_html.status_code == 200
        assert "PeoplePrime Technologies" in res_html.text
        assert "kbGradeVal" in res_html.text
        assert "webGradeVal" in res_html.text
        assert "retryCountVal" in res_html.text
        assert "/static/css/style.css" in res_html.text
        assert "/static/js/app.js" in res_html.text

        # Check static CSS
        res_css = await client.get("/static/css/style.css")
        assert res_css.status_code == 200
        assert ".metrics-grid" in res_css.text

        # Check static JS
        res_js = await client.get("/static/js/app.js")
        assert res_js.status_code == 200
        assert "isSafeUrl" in res_js.text
        assert "/api/documents/ingest" in res_js.text
        assert "/api/chat" in res_js.text


def test_js_safe_url_and_sanitization_rules():
    """Verify that app.js enforces safe protocol and citation rendering."""
    js_content = (BASE_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "isSafeUrl" in js_content
    assert "noopener noreferrer" in js_content
    assert 'adminKeyInput.value = ""' in js_content
    assert "textContent" in js_content


def test_js_markdown_parser_implementation():
    """Verify app.js provides safe DOM-based markdown rendering without raw innerHTML on LLM text."""
    js_content = (BASE_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")

    # Verify parser function definitions
    assert "function renderMarkdownToDOM" in js_content
    assert "function appendInlineMarkdown" in js_content

    # Verify structural markdown elements handled via createElement
    assert 'document.createElement("table")' in js_content
    assert 'document.createElement("thead")' in js_content
    assert 'document.createElement("tbody")' in js_content
    assert 'document.createElement("th")' in js_content
    assert 'document.createElement("td")' in js_content
    assert 'document.createElement("ul")' in js_content
    assert 'document.createElement("ol")' in js_content
    assert 'document.createElement("li")' in js_content
    assert 'document.createElement("blockquote")' in js_content
    assert 'document.createElement("strong")' in js_content
    assert 'document.createElement("em")' in js_content
    assert 'document.createElement("code")' in js_content

    # Verify that raw LLM response text is NOT assigned directly to innerHTML
    # bubble.innerHTML = data.answer should NOT exist
    assert "bubble.innerHTML = data.answer" not in js_content
    assert "bubble.innerHTML = text" not in js_content


def test_css_markdown_styles_defined():
    """Verify style.css provides typography and table styles for rendered markdown inside message bubbles."""
    css_content = (BASE_DIR / "static" / "css" / "style.css").read_text(encoding="utf-8")

    assert ".bubble table" in css_content
    assert ".bubble th" in css_content
    assert ".bubble td" in css_content
    assert ".bubble h1" in css_content
    assert ".bubble ul" in css_content
    assert ".bubble ol" in css_content
    assert ".bubble blockquote" in css_content
    assert ".bubble code" in css_content


def _render_app_js_markdown(markdown_text: str) -> str:
    """Executes the actual renderMarkdownToDOM function from app.js in a headless DOM harness."""
    import json
    import subprocess

    js_path = BASE_DIR / "static" / "js" / "app.js"
    runner_code = f"""
const fs = require('fs');
class Node {{
  constructor(name) {{
    this.tagName = name.toUpperCase();
    this.childNodes = [];
    this.attributes = {{}};
  }}
  appendChild(c) {{ this.childNodes.push(c); return c; }}
  setAttribute(k, v) {{ this.attributes[k] = v; }}
  toString() {{
    if (this.tagName === '#TEXT') return this.textContent || '';
    const attrStr = Object.entries(this.attributes).map(([k,v]) => ` ${{k}}="${{v}}"`).join('');
    return `<${{this.tagName.toLowerCase()}}${{attrStr}}>${{this.childNodes.map(c => c.toString()).join('')}}</${{this.tagName.toLowerCase()}}>`;
  }}
}}
class TextNode {{
  constructor(t) {{ this.tagName = '#TEXT'; this.textContent = t; }}
  toString() {{ return this.textContent || ''; }}
}}
global.document = {{
  createElement: (t) => {{
    const el = new Node(t);
    Object.defineProperty(el, 'href', {{
      set(v) {{ this.setAttribute('href', v); }},
      get() {{ return this.attributes['href']; }}
    }});
    Object.defineProperty(el, 'target', {{
      set(v) {{ this.setAttribute('target', v); }},
      get() {{ return this.attributes['target']; }}
    }});
    Object.defineProperty(el, 'rel', {{
      set(v) {{ this.setAttribute('rel', v); }},
      get() {{ return this.attributes['rel']; }}
    }});
    Object.defineProperty(el, 'textContent', {{
      set(v) {{ this.childNodes = [new TextNode(v)]; }},
      get() {{ return this.childNodes.map(c => c.toString()).join(''); }}
    }});
    return el;
  }},
  createTextNode: (t) => new TextNode(t),
  getElementById: () => null,
  querySelectorAll: () => []
}};
global.window = global;

const appJs = fs.readFileSync({json.dumps(str(js_path))}, 'utf8');
const fn = new Function('document', 'window', 'URL', appJs + '; return {{ renderMarkdownToDOM }};');
const {{ renderMarkdownToDOM }} = fn(global.document, global.window, global.URL);

const container = global.document.createElement('div');
renderMarkdownToDOM(container, {json.dumps(markdown_text)});
console.log(container.toString());
"""
    proc = subprocess.run(["node", "-e", runner_code], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def test_render_inline_bold_annual_leave_sentence():
    """Verify that the actual response-rendering code path turns **bold** into <strong> without raw **."""
    input_text = "**Employees receive 20 paid annual leave days per calendar year.**"
    html_output = _render_app_js_markdown(input_text)

    # Must contain strong tag with exact wording
    assert "<strong>Employees receive 20 paid annual leave days per calendar year.</strong>" in html_output
    # Must NOT have raw ** left in the rendered markup
    assert "**" not in html_output


def test_render_mixed_bold_italic_code_and_quotes():
    """Verify that mixed inline bold, italic, code, and links render properly without interference."""
    input_text = (
        "Employees receive **20 paid annual leave days** per year. "
        "Note: *Subject to manager approval* and `HR_PORTAL`."
    )
    html_output = _render_app_js_markdown(input_text)

    assert "<strong>20 paid annual leave days</strong>" in html_output
    assert "<em>Subject to manager approval</em>" in html_output
    assert "<code>HR_PORTAL</code>" in html_output
    assert "**" not in html_output
    assert "`" not in html_output


def test_render_blocks_lists_and_tables():
    """Verify that block structures (headings, lists, tables) are rendered correctly."""
    input_text = (
        "### Annual Leave Policy\n"
        "- **Annual Leave**: 20 days\n"
        "- **Sick Leave**: 12 days\n\n"
        "| Leave Type | Days |\n"
        "| --- | --- |\n"
        "| Annual | **20** |\n"
    )
    html_output = _render_app_js_markdown(input_text)

    assert "<h3>Annual Leave Policy</h3>" in html_output
    assert "<ul>" in html_output
    assert "<li><strong>Annual Leave</strong>: 20 days</li>" in html_output
    assert "<table>" in html_output
    assert "<th>Leave Type</th>" in html_output
    assert "<td><strong>20</strong></td>" in html_output


def test_render_xss_protection_on_links_and_tags():
    """Verify that malicious javascript: links and tags do not create executable links."""
    input_text = "Check [Malicious](javascript:alert('xss')) and safe [Portal](https://peopleprime.internal)."
    html_output = _render_app_js_markdown(input_text)

    # javascript: link should be neutralized to plain text
    assert 'href="javascript:' not in html_output
    assert "Malicious" in html_output

    # Safe link should have attributes
    assert 'href="https://peopleprime.internal"' in html_output
    assert 'rel="noopener noreferrer"' in html_output
    assert 'target="_blank"' in html_output


def test_render_escaped_bold_markdown():
    """Verify that escaped formatting delimiters like \\*\\* are converted to strong tags without backslashes."""
    input_text = r"\*\*Employees receive 20 paid working days of annual leave per calendar year.\*\*"
    html_output = _render_app_js_markdown(input_text)

    assert "<strong>Employees receive 20 paid working days of annual leave per calendar year.</strong>" in html_output
    assert r"\*" not in html_output
    assert "**" not in html_output


def test_render_math_asterisks_preserved():
    """Verify that literal asterisks in normal mathematical expressions remain literal."""
    input_text = "Calculation: 5 * 10 = 50 and 20 * 2 = 40."
    html_output = _render_app_js_markdown(input_text)

    assert "5 * 10 = 50" in html_output
    assert "20 * 2 = 40" in html_output
    assert "<em>" not in html_output


def test_render_single_line_multiple_table_rows():
    """Verify that single-line tables with multiple pipe-separated rows are parsed into full table markup."""
    input_text = (
        "| Date | Holiday | Day | Category | |---|---|---|---| "
        "| Jan 26 | Republic Day | Monday | Gazetted | "
        "| Aug 15 | Independence Day | Saturday | Gazetted |"
    )
    html_output = _render_app_js_markdown(input_text)

    assert "<table>" in html_output
    assert "<thead>" in html_output
    assert "<th>Holiday</th>" in html_output
    assert "<th>Category</th>" in html_output
    assert "<tbody>" in html_output
    assert "<td>Republic Day</td>" in html_output
    assert "<td>Independence Day</td>" in html_output
    assert "<td>Gazetted</td>" in html_output
    # Must not contain raw divider strings
    assert "|---|---|---|---|" not in html_output
