
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "httpx",
#   "pydantic",
# ]
# ///

import os
import sys
import traceback
from io import StringIO

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, {})

        return {
            "success": True,
            "output": stdout_buffer.getvalue()
            + stderr_buffer.getvalue(),
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc(),
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def analyze_error_with_ai(code: str, error_traceback: str) -> list[int]:
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    prompt = f"""
Analyze the Python code and traceback.
Identify only the line number(s) in the submitted code that caused
the error. Do not include lines from the server or Python libraries.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return JSON only in this format:
{{"error_lines": [3]}}
"""

    response = httpx.post(
        "https://aipipe.org/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON with an error_lines array "
                        "containing only line numbers from the submitted "
                        "Python code. Exclude server and library lines."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=30.0,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return ErrorAnalysis.model_validate_json(content).error_lines


def get_traceback_line_numbers(output: str) -> list[int]:
    lines = []

    for line in output.splitlines():
        marker = 'File "<string>", line '
        if marker in line:
            try:
                number = int(line.split(marker, 1)[1].split(",", 1)[0])
                lines.append(number)
            except ValueError:
                continue

    return lines


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    try:
        error_lines = analyze_error_with_ai(
            request.code, execution["output"]
        )
    except Exception:
        error_lines = get_traceback_line_numbers(execution["output"])

    # Keep only valid line numbers from the submitted code.
    valid_lines = set(range(1, len(request.code.splitlines()) + 1))
    error_lines = sorted({
        line for line in error_lines
        if line in valid_lines
    })

    if not error_lines:
        error_lines = get_traceback_line_numbers(execution["output"])

    return {
        "error": error_lines,
        "result": execution["output"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
