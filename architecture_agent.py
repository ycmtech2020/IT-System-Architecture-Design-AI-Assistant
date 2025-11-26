"""
architecture_agent.py

Core logic for the Architecture Design Assistant.

- Uses Azure OpenAI (via LangChain's ChatOpenAI).
- Wrapped in a LangGraph workflow with an in-memory checkpointer
  (MemorySaver) so the backend keeps per-conversation state.
- Now explicitly REFINES the previous architecture plan (if any)
  instead of redesigning from scratch on follow-up prompts.
"""

import json
from typing import Dict, Any, List, TypedDict, Annotated
import httpx
from langchain_openai import ChatOpenAI
import config
import logging
import traceback
from openai import InternalServerError

import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # later you can change to DEBUG

# ===========================
# AZURE OPENAI CLIENT
# ===========================

# Local-only POC workaround to avoid corporate SSL cert issues
http_client = httpx.Client(verify=False)

client = ChatOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    base_url=config.AZURE_OPENAI_ENDPOINT,
    model=config.OPENAI_MODEL,
    http_client=http_client,
    # Lower temperature to improve determinism / consistency
    temperature=0.1,
)

# ===========================
# Load templates
# ===========================

with open(config.TEMPLATES_PATH, "r", encoding="utf-8") as f:
    TEMPLATE_DATA = json.load(f)


def build_prompt_messages(
    user_message: str,
    previous_arch_plan: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """
    Build the message list for the LLM.

    user_message:
        The full accumulated requirements text
        (first prompt + any refinements appended by the backend).

    previous_arch_plan:
        The last architecture plan produced for this conversation
        (if any). When present, the model is told to REFINE it,
        not redesign from scratch.
    """
    template_summaries = [
        {
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
        }
        for p in TEMPLATE_DATA.get("patterns", [])
    ]

    templates_str = json.dumps(template_summaries, indent=2)

    # ---- SYSTEM PROMPT ----
    system_content = (
        "You are an Architecture Design Assistant for IT systems. "
        "Your job is to take high-level requirements and propose a system architecture.\n\n"
        "You have access to a small library of architecture patterns. "
        "Each pattern has an id, name, and description. Use them as reusable reference designs.\n\n"
        "Return ONLY JSON (no markdown outside the JSON block, no extra text). "
        "The JSON MUST have this structure:\n"
        "{\n"
        "  \"summary\": \"An HTML-formatted architecture summary.\",\n"
        "  \"pattern_id\": \"id of the pattern you are closest to (or 'custom' if none fits)\",\n"
        "  \"components\": [\n"
        "    {\"id\": \"short_id\", \"label\": \"Readable name\", \"type\": \"e.g. web, app, db, cache, queue, mobile_client\"}\n"
        "  ],\n"
        "  \"connections\": [\n"
        "    {\"from\": \"component_id\", \"to\": \"component_id\", \"label\": \"protocol or purpose\"}\n"
        "  ]\n"
        "}\n"
        "IDs must be valid Graphviz node identifiers (letters, digits, underscores only). "
        "Use about 4–12 components to keep the diagram readable.\n\n"
        "IMPORTANT: The `summary` field MUST be valid HTML, not markdown. Use tags like:\n"
        "- <h3>Overview</h3>\n"
        "- <h3>Key Components</h3>\n"
        "- <h3>Data Flow</h3>\n"
        "- <h3>Scalability & Reliability</h3>\n"
        "Within each section, use <ul><li>...</li></ul> bullet lists.\n\n"
        "SUMMARY LENGTH RULES:\n"
        "- Keep the HTML formatting EXACTLY the same (h3 headings + bullet lists).\n"
        "- Keep all <ul><li>...</li></ul> bullet lists.\n"
        "- Make the summary concise: shorten each bullet point using brief, telegraphic text.\n"
        "- Keep the meaning but remove verbosity.\n"
        "- Target 40–60% of the usual summary length.\n\n"
        "REFINEMENT RULES:\n"
        "- If a previous architecture plan is provided, treat it as the BASELINE.\n"
        "- You MUST keep existing component IDs and labels as stable as possible.\n"
        "- Prefer to ADD components or connections rather than renaming or deleting.\n"
        "- Only change or remove existing components if the new requirements clearly conflict.\n"
        "- If a previous pattern_id is provided, keep the same pattern_id unless the user explicitly asks to change the pattern.\n"
    )

    # ---- USER PROMPT ----
    user_content_parts: List[str] = []

    user_content_parts.append("Here are the available architecture patterns:\n")
    user_content_parts.append(templates_str)

    if previous_arch_plan:
        # We are in a follow-up turn; show the previous plan to the model
        user_content_parts.append(
            "\n\nHere is the PREVIOUS architecture plan JSON. "
            "This is your baseline. REFINE this plan instead of redesigning from scratch:\n"
        )
        user_content_parts.append(json.dumps(previous_arch_plan, indent=2))

        user_content_parts.append(
            "\n\nThe user has provided NEW requirements / refinements. "
            "Update the existing architecture minimally to satisfy them:\n"
        )
        user_content_parts.append(user_message)
    else:
        # First turn: design from scratch based on full requirements text
        user_content_parts.append(
            "\n\nThe FULL set of user requirements (including any refinements) is:\n"
        )
        user_content_parts.append(user_message)

    user_content = "".join(user_content_parts)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def _call_model(
    user_message: str,
    previous_arch_plan: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    Internal helper that calls the LLM and parses the JSON architecture plan.

    user_message:
        The full accumulated requirements text built by the backend.

    previous_arch_plan:
        The last architecture plan for this conversation (if any).
        When present, the model is instructed to REFINE it.
    """
    if not config.AZURE_OPENAI_API_KEY:
        raise RuntimeError("Missing Azure OpenAI API key in config.py")

    messages = build_prompt_messages(user_message, previous_arch_plan)

    print("=== FULL REQUIREMENTS SENT TO MODEL ===")
    print(user_message)
    print("=== PREVIOUS ARCH PLAN (if any) ===")
    print(json.dumps(previous_arch_plan, indent=2) if previous_arch_plan else "None")
    print("=======================================")

    # Build a single prompt string for ChatOpenAI (as in your original version)
    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    full_prompt = system_content + "\n\n" + user_content

    try:
        # ChatOpenAI interface: use invoke()
        llm_result = client.invoke(full_prompt)

        # llm_result is a ChatMessage-like object; get the text content
        raw_text = getattr(llm_result, "content", str(llm_result))

        print("=== RAW MODEL OUTPUT ===")
        print(raw_text)
        print("========================")

        # ---- CLEANUP LOGIC ----
        import re

        clean_text = raw_text.strip()

        # If the model wrapped JSON in ```json ... ```
        if clean_text.startswith("```"):
            # Extract the first {...} block from inside
            match = re.search(r"\{[\s\S]*\}", clean_text)
            if match:
                clean_text = match.group(0)

        print("=== CLEAN JSON CANDIDATE ===")
        print(clean_text)
        print("============================")

        try:
            arch_plan = json.loads(clean_text)  # ✅ use cleaned text
        except Exception as e:
            print("JSON parse failed, using fallback architecture.")
            print("JSON error:", e)
            arch_plan = _fallback_architecture("Could not parse JSON from model output.")

    except InternalServerError as e:
        # This is a 5xx from your gateway (genailab.tcs.in)
        logger.error("Azure gateway returned 500. Status: %s", e.status_code)
        try:
            logger.error("Response body: %s", e.response.text)
        except Exception:
            pass
        raise RuntimeError(
            "Server error from genailab.tcs.in (500). Check gateway logs / configuration."
        ) from e

    except Exception as ex:
        # 🔍 Detailed logging
        logger.error("Azure OpenAI call failed: %s", ex)
        logger.error("Exception type: %s", type(ex).__name__)
        logger.error("Full traceback:\n%s", traceback.format_exc())
        raise RuntimeError(
            "Connection error: unable to reach Azure OpenAI. Please check network / VPN."
        ) from ex

    # Ensure keys exist
    arch_plan.setdefault("summary", "No summary provided.")
    arch_plan.setdefault("pattern_id", "unknown")
    arch_plan.setdefault("components", [])
    arch_plan.setdefault("connections", [])

    print("=== PARSED ARCH PLAN ===")
    print(json.dumps(arch_plan, indent=2))
    print("========================")

    return arch_plan


def _fallback_architecture(reason: str) -> Dict[str, Any]:
    """
    Fallback architecture plan when something goes wrong.
    """
    return {
        "summary": (
            f"Fallback architecture used because: {reason}\n\n"
            "This is a simple three-tier web application."
        ),
        "pattern_id": "fallback_three_tier",
        "components": [
            {"id": "client", "label": "Client", "type": "client"},
            {"id": "web", "label": "Web Server", "type": "web"},
            {"id": "app", "label": "Application Server", "type": "app"},
            {"id": "db", "label": "Database", "type": "database"},
        ],
        "connections": [
            {"from": "client", "to": "web", "label": "HTTP/HTTPS"},
            {"from": "web", "to": "app", "label": "Internal HTTP"},
            {"from": "app", "to": "db", "label": "SQL"},
        ],
    }


# ===========================
# LangGraph: state + memory
# ===========================

class ArchState(TypedDict):
    """
    State for the LangGraph workflow.

    - messages: list of requirement text snapshots
      (we append each call's full requirements string here so we
       always know the latest).
    - arch_plan: the latest parsed architecture JSON from the model.
    - arch_history: list of ALL architecture plans produced so far
      for this conversation (used to get the previous plan on follow-ups).
    """
    messages: Annotated[List[str], operator.add]
    arch_plan: Dict[str, Any]
    arch_history: Annotated[List[Dict[str, Any]], operator.add]


def _llm_node(state: ArchState) -> ArchState:
    """
    LangGraph node that calls the model using the latest requirements text.

    The MemorySaver checkpointer keeps the 'messages' and 'arch_history'
    per thread_id, so follow-up turns can refine the previous architecture.
    """
    messages = state.get("messages") or []
    if not messages:
        raise RuntimeError("No requirements text provided to LLM node.")

    # Latest full requirements text (first prompt + refinements)
    latest_requirements = messages[-1]

    arch_history = state.get("arch_history") or []
    previous_arch_plan = arch_history[-1] if arch_history else None

    arch_plan = _call_model(latest_requirements, previous_arch_plan)

    # Return only the NEW plan as a delta for arch_history.
    # MemorySaver + operator.add will append it to the stored list.
    return {
        "messages": [],             # we've consumed messages for this step
        "arch_plan": arch_plan,     # latest plan
        "arch_history": [arch_plan] # append-only history
    }


# Build the LangGraph workflow with in-memory checkpointing
_graph_builder = StateGraph(ArchState)
_graph_builder.add_node("llm", _llm_node)
_graph_builder.set_entry_point("llm")
_graph_builder.add_edge("llm", END)

_checkpointer = MemorySaver()
_arch_graph = _graph_builder.compile(checkpointer=_checkpointer)


def call_llm_for_architecture(
    user_message: str,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """
    Public entry point used by the Flask app.

    user_message:
        The full requirements text (first prompt + refinements), as built
        by app.py for backward-compatible behavior.

    thread_id:
        Per-conversation identifier used by LangGraph's MemorySaver
        to maintain server-side state across turns. Should come from
        the frontend's conversation_id.
    """
    if not config.AZURE_OPENAI_API_KEY:
        raise RuntimeError("Missing Azure OpenAI API key in config.py")

    initial_state: ArchState = {
        "messages": [user_message],
        "arch_plan": {},
        "arch_history": [],
    }

    final_state = _arch_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    arch_plan = final_state.get("arch_plan") or _fallback_architecture(
        "Missing arch_plan from LangGraph state."
    )
    return arch_plan
