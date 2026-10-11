
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

        # Compile separately so syntax errors retain their line number.
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
Analyze the Python code and traceback below.
Return the exact line number(s) in the submitted code responsible
for the error. Use the line number shown in the traceback.
Return only JSON matching this schema:
{{"error_lines": [3]}}

CODE:
{code}

TRACEBACK:
{error_traceback}
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
                        "You identify Python error line numbers. "
                        "Return valid JSON with an error_lines array "
                        "of integers. Do not invent line numbers."
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
        # Fallback: recover the line number from Python's traceback.
        # This keeps the endpoint usable if the AI service is unavailable.
        error_lines = []
        for line in execution["output"].splitlines():
            if 'line ' in line and ', in ' in line:
                try:
                    error_lines.append(
                        int(line.split('line ')[1].split(',')[0])
                    )
                except ValueError:
                    pass

        if not error_lines:
            error_lines = [1]

    return {"error": error_lines, "result": execution["output"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
