"""Render structural-pattern trees without implying a grammatical parse."""

from html import escape

from linear_a.grammar.pcfg_engine import ParseNode


def render_ascii_tree(node: ParseNode, prefix: str = "", is_tail: bool = True) -> str:
    """Return a plain-text structural-pattern tree."""
    connector = "└── " if is_tail else "├── "
    text = f' "{node.text}"' if node.text else ""
    output = f"{prefix}{connector}{node.symbol}{text}\n"
    child_prefix = prefix + ("    " if is_tail else "│   ")
    for index, child in enumerate(node.children):
        output += render_ascii_tree(child, child_prefix, index == len(node.children) - 1)
    return output


def render_html_tree(node: ParseNode) -> str:
    """Return escaped nested HTML for a structural-pattern tree."""
    text = f" <strong>{escape(node.text)}</strong>" if node.text else ""
    children = "".join(render_html_tree(child) for child in node.children)
    return (
        '<div class="syntax-node" style="margin-left:18px;border-left:2px solid var(--border);padding-left:10px;margin-top:6px;">'
        f'<div style="font-size:12px;font-weight:600;color:var(--accent-indigo);">{escape(node.symbol)}{text}</div>{children}</div>'
    )
