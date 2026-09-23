"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


GATEWAY_URL = "https://customersupportgateway-luqui0v9kz.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp" 
KB_ID       = "XW0VAAOSYB"          
REGION      = "us-east-1"        
MEMORY_ID   = "CustomerSupportMemory-115LawCviY"        


model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)

memory_client = MemoryClient(region_name=REGION)

system_prompt = """
You are an e-commerce customer support assistant.
Use the available tools to answer customer questions accurately.

When calculate_loyalty_discount is used, the values returned by the tool are
authoritative. Copy all calculated values from the tool result exactly.
Do not independently calculate, verify, correct, or override any values.

In particular, points discounts and tier discounts are sequential, not
independent. The tier discount is applied to the subtotal remaining after
the points discount, not to the original order total.

If the tool returns values for points_redeemed, points_discount,
tier_discount, final_total, total_savings, points_earned, or
remaining_points, report those exact values to the customer.
"""

_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}

    for strategy in strategies:
        strategy_type = strategy["type"]

        if strategy.get("namespaceTemplates"):
            namespace = strategy["namespaceTemplates"][0]
        elif strategy.get("namespaces"):
            namespace = strategy["namespaces"][0]
        else:
            continue

        namespaces[strategy_type] = namespace

    return namespaces


class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        messages = event.agent.messages
        if not messages:
            return

        last_message = messages[-1]

        # Only process user messages
        if last_message.get("role") != "user":
            return

        content = last_message.get("content", [])

        # Ignore tool-result messages and find plain text
        if not content or any("toolResult" in block for block in content):
            return

        query = next(
            (block.get("text") for block in content if block.get("text")),
            None,
        )

        if not query:
            return

        memory_texts = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)

            memories = self.memory_client.retrieve_memories(
                self.memory_id,
                namespace,
                query,
                top_k=5,
            )

            for memory in memories:
                text = memory.get("content", {}).get("text")
                if text:
                    memory_texts.append(f"[{strategy_type}] {text}")

        if memory_texts:
            context = "\n".join(memory_texts)
            last_message["content"] = [
                {
                    "text": f"Customer Context:\n{context}\n\n{query}"
                }
            ]

    def save_support_interaction(self, event: AfterInvocationEvent):
        messages = event.agent.messages

        customer_query = None
        agent_response = None

        for message in reversed(messages):
            role = message.get("role")
            content = message.get("content", [])

            # Skip tool-result messages
            if any("toolResult" in block for block in content):
                continue

            text = next(
                (block.get("text") for block in content if block.get("text")),
                None,
            )

            if not text:
                continue

            if role == "assistant" and agent_response is None:
                agent_response = text

            elif role == "user" and customer_query is None:
                customer_query = text

            if customer_query and agent_response:
                break

        if customer_query and agent_response:
            self.memory_client.create_event(
                self.memory_id,
                self.actor_id,
                self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (agent_response, "ASSISTANT"),
                ],
            )

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context,
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    resp = _bedrock_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
    )

    results = resp.get("retrievalResults", [])

    if not results:
        return "No relevant information found in the knowledge base."

    chunks = []

    for result in results:
        text = result.get("content", {}).get("text")
        if text:
            chunks.append(text)

    if not chunks:
        return "No relevant information found in the knowledge base."

    return "\n---\n".join(chunks)

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json
import math

loyalty_points = {loyalty_points}
tier = {tier!r}
order_total = {order_total}
product_category = {product_category!r}

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5
}}

tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15
}}

# Redeem points in increments of 500.
# 500 points = $5, with redemption capped at 50% of the order total.
max_points_value = order_total * 0.50
available_blocks = loyalty_points // 500
max_blocks = math.floor(max_points_value / 5)
blocks_redeemed = min(available_blocks, max_blocks)

points_redeemed = blocks_redeemed * 500
points_discount = blocks_redeemed * 5

subtotal_after_points = order_total - points_discount

tier_rate = tier_rates.get(tier, 0.00)
tier_discount = subtotal_after_points * tier_rate

final_total = subtotal_after_points - tier_discount
total_savings = points_discount + tier_discount

earn_rate = earn_rates.get(product_category, 1)
points_earned = math.floor(final_total * earn_rate)

remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier_discount_pct": int(tier_rate * 100),
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

            for event in response["stream"]:
                return json.dumps(event)

        return json.dumps({"error": "No result returned from Code Interpreter"})

    except Exception as e:
        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        tier_rate = tier_rates.get(tier, 0.00)
        tier_discount = order_total * tier_rate
        final_total = order_total - tier_discount

        return json.dumps({
            "points_redeemed": 0,
            "tier_discount_pct": int(tier_rate * 100),
            "tier_discount": round(tier_discount, 2),
            "final_total": round(final_total, 2),
            "remaining_points": loyalty_points,
            "fallback": True,
            "error": str(e),
        })

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    user_input = payload.get("prompt", "")
    actor_id = payload.get("customer_id", "anonymous")
    session_id = payload.get("session_id") or str(uuid.uuid4())

    memory_hook = MemoryHook(
        actor_id=actor_id,
        session_id=session_id,
        memory_client=memory_client,
        memory_id=MEMORY_ID,
    )

    agent_core_browser = AgentCoreBrowser(region=REGION)

    tools = [
        search_knowledge_base,
        calculate_loyalty_discount,
        agent_core_browser.browser,
    ]

    mcp_client = MCPClient(
        lambda: streamable_http_client(GATEWAY_URL)
    )

    try:
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=system_prompt,
            )

            response = agent(user_input)
            return response.message["content"][0]["text"]

    except Exception as e:
        logger.error(f"Agent invocation failed: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()
