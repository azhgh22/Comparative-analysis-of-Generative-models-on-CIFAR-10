import json
from typing import List, Dict, Any


def parse_notebook(json_data: str | dict) -> List[Dict[str, Any]]:
    """
    Parse Jupyter notebook JSON and extract code and markdown cells.

    Args:
        json_data: Either a JSON string or a parsed dict

    Returns:
        List of dicts with 'type' and 'content' keys
    """
    if isinstance(json_data, str):
        notebook = json.loads(json_data)
    else:
        notebook = json_data

    cells = []

    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")

        if cell_type in ("code", "markdown"):
            # Source can be a string or list of strings
            source = cell.get("source", [])
            if isinstance(source, list):
                content = "".join(source)
            else:
                content = source

            cells.append({
                "type": cell_type,
                "content": content.strip()
            })

    return cells


def extract_code_only(json_data: str | dict) -> List[str]:
    """Extract only code cells from notebook."""
    cells = parse_notebook(json_data)
    return [c["content"] for c in cells if c["type"] == "code"]


def extract_markdown_only(json_data: str | dict) -> List[str]:
    """Extract only markdown cells from notebook."""
    cells = parse_notebook(json_data)
    return [c["content"] for c in cells if c["type"] == "markdown"]


if __name__ == "__main__":
    # Example usage
    with open("nt.txt", "r") as f:
        notebook_json = json.load(f)

    parsed = parse_notebook(notebook_json)

    for i, cell in enumerate(parsed):
        print(f"--- Cell {i + 1} ({cell['type']}) ---")
        print(cell["content"])
        print()
