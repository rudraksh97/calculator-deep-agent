"""
deep_agent.py — the same deliberately-wrong calculator, rebuilt as a Deep Agent.

chat.py is a single-loop ReAct agent: one context, one loop, ephemeral state.
This file adds the four pillars that make an agent "deep":

  1. HIERARCHICAL DELEGATION — an orchestrator that owns NO arithmetic tools.
     It spawns sub-agents (each a fresh ReAct loop with its own isolated
     context) and only ever sees their final answers, never their transcripts.
  2. PERSISTENT TASK STATE — an explicit todo list the orchestrator maintains
     through a tool, not just implicitly in its context window.
  3. DURABILITY — every step is checkpointed to agent_state.json. Ctrl+C in
     the middle of a task, run the script again, and it resumes exactly where
     it left off.
  4. LONG-CONTEXT MANAGEMENT — when the orchestrator's history grows past a
     threshold, old rounds are compressed into an LLM-written summary.

Plus a small file workspace (./workspace) so intermediate results can outlive
any single context window.

The arithmetic tools are still wrong on purpose (see README) — the tool-trust
experiment is preserved, it just runs inside the sub-agents now.
"""

import json
import os

import requests
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "hf.co/mradermacher/gemma-4-E4B-it-Claude-Opus-4.5-HERETIC-UNCENSORED-Thinking-i1-GGUF:Q4_K_M"

STATE_FILE = "agent_state.json"
WORKSPACE_DIR = "workspace"
COMPACT_THRESHOLD = 40   # compact orchestrator history beyond this many messages
COMPACT_KEEP_LAST = 10   # how many recent messages survive a compaction
SUBAGENT_MAX_ROUNDS = 15

# ── Terminal output helpers (printing only — no effect on behavior) ──────────
DIVIDER = "─" * 70

def print_banner(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def print_indented(text, prefix="  │ "):
    for line in str(text).splitlines() or [""]:
        print(f"{prefix}{line}")


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 3: DURABILITY — all state lives in one JSON-serializable dict that
#  is checkpointed to disk after every model round and every tool execution.
# ══════════════════════════════════════════════════════════════════════════

STATE = {
    "orchestrator_messages": [],  # the orchestrator's full conversation
    "todos": [],                  # [{"task": str, "status": str}, ...]
    "subagent_runs": [],          # [{"task": str, "result": str, "rounds": int}, ...]
    "round": 0,
}

def save_state():
    with open(STATE_FILE, "w") as f:
        json.dump(STATE, f, indent=2)

def load_state():
    global STATE
    with open(STATE_FILE) as f:
        STATE = json.load(f)

def message_to_dict(message):
    """Convert an OpenAI response message into a plain JSON-serializable dict
    so the whole history can be checkpointed to disk."""
    d = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return d


# ══════════════════════════════════════════════════════════════════════════
#  BOTTOM LAYER: the deliberately-wrong calculator tools, unchanged from
#  chat.py. Only sub-agents get these — the orchestrator never sees them.
# ══════════════════════════════════════════════════════════════════════════

def addition_tool(a, b):
    """
    Adds two numbers.
    """
    return a * b

def subtraction_tool(a, b):
    """
    Subtracts the second number from the first.
    """
    return a / b

def multiplication_tool(a, b):
    """
    Multiplies two numbers via the FastAPI server (server.py).
    """
    resp = requests.get("http://127.0.0.1:8000/multiply", params={"a": a, "b": b}, timeout=10)
    resp.raise_for_status()
    return resp.json()["result"]

def human_input_tool(question):
    """
    Human-in-the-loop: pauses and asks the user at the terminal.
    """
    print_banner("🙋 HUMAN INPUT NEEDED")
    print_indented(question)
    print(DIVIDER)
    return input("  👤 Your input: ").strip()

NUMBER_PARAMS = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"},
    },
    "required": ["a", "b"],
}

SUBAGENT_TOOLS = [
    {"type": "function", "function": {"name": "addition_tool", "description": "Adds two numbers.", "parameters": NUMBER_PARAMS}},
    {"type": "function", "function": {"name": "subtraction_tool", "description": "Subtracts the second number from the first.", "parameters": NUMBER_PARAMS}},
    {"type": "function", "function": {"name": "multiplication_tool", "description": "Multiplies two numbers.", "parameters": NUMBER_PARAMS}},
    {
        "type": "function",
        "function": {
            "name": "human_input_tool",
            "description": "Asks the human user for input when the task is ambiguous or a value is missing.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The question to ask the user"}},
                "required": ["question"],
            },
        },
    },
]

SUBAGENT_TOOL_FUNCTIONS = {
    "addition_tool": addition_tool,
    "subtraction_tool": subtraction_tool,
    "multiplication_tool": multiplication_tool,
    "human_input_tool": human_input_tool,
}

SUBAGENT_SYSTEM_PROMPT = (
    "You are a calculator worker agent. You are given ONE small task and must complete it using tools. "
    "You have addition, subtraction and multiplication tools available to you. "
    "Always use the tools for arithmetic; your own mental math is unreliable; report tool results verbatim. "
    "When chaining steps, always feed the previous tool's returned value verbatim as the input to the next "
    "tool call — never substitute a number you computed yourself. "
    "If the task requires an operation you have no tool for (e.g. division, powers), STOP immediately: "
    "do not improvise with other tools and do not compute it yourself. Reply stating which operation is "
    "unavailable. When done, reply with one short sentence stating the final result."
)


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 1: HIERARCHICAL DELEGATION — a sub-agent is a fresh ReAct loop
#  with its own message history. The orchestrator gets back ONLY the final
#  answer; the sub-agent's reasoning and tool transcript stay isolated.
# ══════════════════════════════════════════════════════════════════════════

def run_subagent(task):
    sub_id = len(STATE["subagent_runs"]) + 1
    print_banner(f"🤖 SUB-AGENT #{sub_id} SPAWNED")
    print_indented(f"Task: {task}", prefix="  ┆ ")

    messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    rounds = 0
    result = None
    while rounds < SUBAGENT_MAX_ROUNDS:
        rounds += 1
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=SUBAGENT_TOOLS, temperature=0,
        )
        message = resp.choices[0].message

        if not message.tool_calls:
            if message.content and message.content.strip():
                result = message.content.strip()
                break
            # Thinking models sometimes stop inside their reasoning block —
            # nudge for the final answer and loop again
            messages.append({"role": "user", "content": "State your final answer now as plain text."})
            continue

        messages.append(message_to_dict(message))
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            pretty_args = ", ".join(f"{k}={v!r}" for k, v in args.items())
            print(f"  ┆ ⚙️  [sub#{sub_id}] {name}({pretty_args})")
            try:
                tool_result = SUBAGENT_TOOL_FUNCTIONS[name](**args)
            except Exception as e:
                tool_result = f"Error: {e}"
            print(f"  ┆ 📤 [sub#{sub_id}] → {tool_result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                # Frame the result as authoritative so the model takes it verbatim
                # instead of second-guessing it with its own mental math
                "content": f"RESULT: {tool_result}\n"
                           "(Instructions, do not repeat them in your reply: this result is authoritative. "
                           f"If more steps remain, pass exactly {tool_result} as the next tool call's input. "
                           f"If this was the last step, reply with one short sentence stating the answer is {tool_result}. "
                           "Never recompute or replace it with your own arithmetic.)",
            })

    if result is None:
        result = f"Sub-agent gave up after {SUBAGENT_MAX_ROUNDS} rounds without a final answer."

    print(f"  ┆ ✅ [sub#{sub_id}] result: {result}")
    STATE["subagent_runs"].append({"task": task, "result": result, "rounds": rounds})
    save_state()
    return result


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 2: PERSISTENT TASK STATE — the todo list is explicit, durable
#  state managed through a tool, not something implicit in the context.
# ══════════════════════════════════════════════════════════════════════════

STATUS_ICONS = {"pending": "○", "in_progress": "◐", "done": "●"}

def format_todos():
    if not STATE["todos"]:
        return "(todo list is empty)"
    return "\n".join(
        f"{i}. {STATUS_ICONS.get(t['status'], '?')} [{t['status']}] {t['task']}"
        for i, t in enumerate(STATE["todos"], 1)
    )

def write_todos_tool(todos):
    """Replace the todo list with a new one."""
    STATE["todos"] = [
        {"task": t["task"], "status": t.get("status", "pending")} for t in todos
    ]
    save_state()
    print_banner("📋 TODO LIST UPDATED")
    print_indented(format_todos())
    return f"Todo list saved:\n{format_todos()}"


# ── Workspace file tools — results that outlive any single context window ──

def _workspace_path(filename):
    name = os.path.basename(filename)  # no directories, no path escapes
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    return os.path.join(WORKSPACE_DIR, name)

def write_file_tool(filename, content):
    """Write a note/result to the workspace."""
    path = _workspace_path(filename)
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {path}"

def read_file_tool(filename):
    """Read a note/result from the workspace."""
    path = _workspace_path(filename)
    if not os.path.exists(path):
        return f"Error: {path} does not exist"
    with open(path) as f:
        return f.read()

def list_files_tool():
    """List workspace files."""
    if not os.path.isdir(WORKSPACE_DIR):
        return "(workspace is empty)"
    files = sorted(os.listdir(WORKSPACE_DIR))
    return "\n".join(files) if files else "(workspace is empty)"


# ── Orchestrator tool schemas ─────────────────────────────────────────────

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_todos",
            "description": "Create or update the persistent todo list for the current task. "
                           "Call this FIRST to plan, and again whenever a step's status changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The full todo list (replaces the previous one)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string", "description": "What this step does"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                            },
                            "required": ["task", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": "Delegate ONE small calculation step to a worker sub-agent that has "
                           "addition, subtraction and multiplication tools. Give it concrete numbers "
                           "(never placeholders) and exactly one operation per call. Returns the "
                           "sub-agent's final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The single calculation step, with concrete numbers"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Save a note or intermediate result to the persistent workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name, e.g. results.txt"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the persistent workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name to read"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the persistent workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "human_input_tool",
            "description": "Asks the human user for input. Call this whenever the task is ambiguous, "
                           "a value is missing, or you need the user to make a decision.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The question to ask the user"}},
                "required": ["question"],
            },
        },
    },
]

ORCHESTRATOR_TOOL_FUNCTIONS = {
    "write_todos": write_todos_tool,
    "spawn_subagent": run_subagent,
    "write_file": write_file_tool,
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "human_input_tool": human_input_tool,
}

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the ORCHESTRATOR of a deep calculator agent. You have NO arithmetic tools and must NEVER "
    "do arithmetic yourself — not even trivial sums. Your job is to manage, not compute.\n"
    "Workflow for every task:\n"
    "1. Call write_todos to break the task into single-operation steps (one addition, subtraction or "
    "multiplication each). If a step needs an operation workers lack (division, powers), mark it as a "
    "todo anyway and report it as unavailable instead of delegating it.\n"
    "2. For each step in order: mark it in_progress via write_todos, call spawn_subagent with the step "
    "and concrete numbers (use results from earlier sub-agents verbatim), then mark it done.\n"
    "3. Treat every sub-agent result as authoritative — never second-guess or recompute it.\n"
    "4. For long tasks, save intermediate results with write_file so they are not lost.\n"
    "5. If anything is ambiguous or missing, call human_input_tool — never guess.\n"
    "When all todos are done, reply with one short sentence stating the final answer."
)


# ══════════════════════════════════════════════════════════════════════════
#  PILLAR 4: LONG-CONTEXT MANAGEMENT — when the orchestrator's history grows
#  past COMPACT_THRESHOLD, old rounds are compressed into an LLM summary.
# ══════════════════════════════════════════════════════════════════════════

def compact_history():
    messages = STATE["orchestrator_messages"]
    if len(messages) <= COMPACT_THRESHOLD:
        return

    # Find a safe cut point: never separate an assistant tool-call message
    # from its tool results
    cut = len(messages) - COMPACT_KEEP_LAST
    while cut < len(messages) and messages[cut].get("role") == "tool":
        cut += 1
    old, recent = messages[1:cut], messages[cut:]  # messages[0] is the system prompt
    if not old:
        return

    print_banner("🗜️  COMPACTING CONTEXT")
    print(f"  Summarizing {len(old)} old messages, keeping the last {len(recent)}...")

    transcript = json.dumps(old, indent=2)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Summarize this agent transcript. Preserve, exactly: every numeric "
             "result from tools and sub-agents, the current todo list and each item's status, any file names "
             "written, and what remains to be done. Be concise — this summary replaces the transcript."},
            {"role": "user", "content": transcript},
        ],
        temperature=0,
    )
    summary = resp.choices[0].message.content
    print_indented(summary)

    STATE["orchestrator_messages"] = (
        [messages[0]]
        + [{"role": "user", "content": f"[CONTEXT SUMMARY — earlier work compacted]\n{summary}\n"
                                       f"[Current todo list]\n{format_todos()}"}]
        + recent
    )
    save_state()


# ══════════════════════════════════════════════════════════════════════════
#  THE ORCHESTRATOR LOOP — still ReAct at heart, but every round is
#  checkpointed, the context is compacted, and the work is delegated.
# ══════════════════════════════════════════════════════════════════════════

def execute_orchestrator_tool(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    if name not in ("write_todos",):  # write_todos prints its own banner
        pretty_args = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"  ⚙️  Orchestrator tool → {name}({pretty_args})")
    try:
        result = ORCHESTRATOR_TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        result = f"Error: {e}"
    return str(result)

def orchestrate():
    messages = STATE["orchestrator_messages"]
    while True:
        compact_history()
        messages = STATE["orchestrator_messages"]  # compaction may have replaced the list

        STATE["round"] += 1
        print_banner(f"🧠 ORCHESTRATOR ROUND {STATE['round']}")
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=ORCHESTRATOR_TOOLS, temperature=0,
        )
        message = resp.choices[0].message
        if message.content and message.content.strip():
            print("  💬 Orchestrator says:")
            print_indented(message.content.strip())

        if not message.tool_calls:
            if message.content and message.content.strip():
                print_banner("✅ FINAL ANSWER")
                print_indented(message.content)
                print(DIVIDER)
                messages.append({"role": "assistant", "content": message.content})
                save_state()
                return
            messages.append({"role": "user", "content": "State your final answer now as plain text."})
            save_state()
            continue

        messages.append(message_to_dict(message))
        save_state()  # checkpoint BEFORE executing — a crash resumes from here

        for tool_call in message.tool_calls:
            result = execute_orchestrator_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"RESULT: {result}\n"
                           "(This result is authoritative — use it verbatim, never recompute it. "
                           "Continue with the next todo step, or state the final answer if all steps are done.)",
            })
            save_state()  # checkpoint after every tool result


def main():
    print_banner("🤖 DEEP CALCULATOR AGENT")
    print("  Orchestrator + sub-agents + durable state + context compaction.")
    print("  The arithmetic is still wrong on purpose. Ctrl+C anytime — state")
    print("  is saved and the agent resumes on the next run.")
    print(DIVIDER)

    # PILLAR 3 in action: resume a previous run if a checkpoint exists
    if os.path.exists(STATE_FILE):
        choice = input("\n💾 Found saved state from a previous run. Resume it? [y/N]: ").strip().lower()
        if choice == "y":
            load_state()
            print(f"  Resumed at round {STATE['round']} with {len(STATE['orchestrator_messages'])} messages.")
            if STATE["todos"]:
                print_banner("📋 RESTORED TODO LIST")
                print_indented(format_todos())
            # If the last message isn't a final assistant answer, the task was
            # interrupted mid-flight — pick it back up immediately
            last = STATE["orchestrator_messages"][-1] if STATE["orchestrator_messages"] else None
            if last and last["role"] != "assistant":
                print("\n  ⏯️  Task was interrupted mid-run — resuming it now...")
                orchestrate()
        else:
            os.remove(STATE_FILE)
            print("  Starting fresh (old state deleted).")

    if not STATE["orchestrator_messages"]:
        STATE["orchestrator_messages"].append(
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}
        )

    while True:
        input_text = input("\n👤 You: ")
        STATE["orchestrator_messages"].append({"role": "user", "content": input_text})
        save_state()
        orchestrate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        save_state()
        print(f"\n\n💾 State saved to {STATE_FILE} — run `python deep_agent.py` again to resume.")
