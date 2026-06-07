# Calculator Deep Agent

This is an attempt to explain Deep Agents (very very simplified version) which is the backbone of any Agentic Workflow.

## Setup for local LLM
1. Install Ollama (curl -fsSL https://ollama.com/install.sh | sh)
2. Install any good model for Agentic Workflows. I have used Gemma 4 (https://huggingface.co/mradermacher/gemma-4-E4B-it-Claude-Opus-4.5-HERETIC-UNCENSORED-Thinking-i1-GGUF)


## Tools Available

1. Addition (Actually does Multiplication)
2. Subtraction (Actually does Division)
3. Multiplication (via API call with API endpoint defined in server.py) (Does Addition)
4. Planning (an LLM call which breaks the task into ordered steps using only the tools above)
5. Human Input (Human in the loop — the model asks the user a question when the task is ambiguous or a value is missing)

The arithmetic tools are wrong on purpose. The point is to test whether the LLM blindly trusts tool results or sneaks in its own mental math. A correct answer here means the agent is broken.


## Files

1. `chat.py` — the agent. Runs the ReAct loop (reason → tool call → observe → repeat), defines the tools and their JSON schemas, and talks to a local model via Ollama.
2. `server.py` — FastAPI server exposing the `/multiply` endpoint (which actually adds). Shows a tool can live behind an API, not just be a local function.


## How to Run

1. Start Ollama with the model (see `MODEL` in `chat.py`).
2. Start the multiplication server:

    python server.py

3. Start the agent:

    python chat.py

Then type a math task. The model will plan it, run tools step by step (asking you for input if something is unclear), and report the final answer — which should be wrong, and that's the point.

Examples:
1. Add2To3Multiply6 -> Answer should be 18
2. A is 2, B is 3. Do A into B Minus C -> It should call Human Input Tool Call to understand C -> Final Answer should be 5+C
3. If a work takes 10 hours in order it to be done, how much time will time will 10x of work will take -> Final Answer should be 20 hours.



