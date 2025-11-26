"""
app.py

Flask web application that exposes:

- "/"          : Simple UI (HTML) for chatting with the assistant.
- "/api/chat"  : POST endpoint for architecture generation.

Flow:
1. User enters requirements in the browser.
2. Frontend sends the text to /api/chat.
3. Backend:
   - Builds a full requirements text (initial + refinements).
   - Calls the LLM (architecture_agent.call_llm_for_architecture).
   - That function now uses LangGraph + MemorySaver to REFINE
     the previous architecture plan for the same conversation_id.
   - Builds a diagram with Graphviz (diagram_generator.generate_graphviz_diagram).
   - Returns summary, components, connections, SVG URL, and DOT source.
4. Frontend displays the answer and the diagram, and shows DOT code.
"""

from flask import Flask, jsonify, render_template, request

from architecture_agent import call_llm_for_architecture
from diagram_generator import generate_graphviz_diagram

app = Flask(__name__)


@app.route("/")
def index():
    """
    Render the main UI.
    """
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Accepts JSON:
    {
        "message": "latest user message",
        "history": ["previous prompt 1", "previous prompt 2", ...],  # optional
        "conversation_id": "stable id for this conversation (optional)"
    }

    Backend will:
    - Combine history + latest message into one 'full requirements' text.
    - Call call_llm_for_architecture(full_requirements_text, thread_id=conversation_id),
      which now REFINES the previous architecture (if any) for this conversation.
    """
    data = request.get_json(force=True)

    # Latest user message
    user_message = (data.get("message") or "").strip()

    # History from client (list of previous prompts)
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []

    # Conversation identifier for LangGraph thread_id
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        # Fallback: per-client IP if no explicit ID was sent
        conversation_id = request.remote_addr or "default"

    if not user_message and not history:
        return jsonify({"error": "Message is required."}), 400

    # Build full requirements text.
    # This keeps your original refinement semantics:
    # - First prompt is treated as base.
    # - Later prompts are prefixed with "Refinement request:".
    parts = []
    for idx, msg in enumerate(history):
        msg = (msg or "").strip()
        if not msg:
            continue
        if idx == 0:
            # First prompt as-is
            parts.append(msg)
        else:
            # Subsequent prompts marked as refinement
            parts.append(f"Refinement request: {msg}")

    if user_message:
        # Current message as latest refinement (or first if none before)
        if not parts:
            parts.append(user_message)
        else:
            parts.append(f"Refinement request: {user_message}")

    full_requirements_text = "\n\n".join(parts).strip()

    try:
        # Now truly stateful: the LangGraph workflow will use the stored
        # arch_history for this conversation_id to refine the previous plan.
        arch_plan = call_llm_for_architecture(
            full_requirements_text,
            thread_id=conversation_id,
        )
    except RuntimeError as ex:
        # This will catch our "Connection error: unable to reach Azure OpenAI..."
        # from architecture_agent
        return jsonify({"error": str(ex)}), 502  # 502 Bad Gateway (upstream service error)

    image_url, dot_source = generate_graphviz_diagram(arch_plan)

    response_payload = {
        "summary": arch_plan.get("summary", ""),
        "pattern_id": arch_plan.get("pattern_id", ""),
        "components": arch_plan.get("components", []),
        "connections": arch_plan.get("connections", []),
        "image_url": image_url,
        "dot": dot_source,
    }

    return jsonify(response_payload)


if __name__ == "__main__":
    # Run development server
    # Access it in your browser at: http://127.0.0.1:5000
    app.run(debug=True)
