import subprocess
from pathlib import Path
from porsche_agent.tools import tool


@tool(description="Execute a shell command and return stdout/stderr")
def shell_command(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=30
    )
    output = result.stdout
    if result.stderr:
        output += "\n[stderr]\n" + result.stderr
    return output.strip() or "(no output)"


@tool(description="Read the contents of a file at the given path")
def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: file not found: {path}"
    return p.read_text()


@tool(description="Write content to a file, overwriting if it exists")
def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} bytes to {path}"
