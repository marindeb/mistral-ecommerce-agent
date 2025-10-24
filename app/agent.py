"""
Module: agent.py
----------------
Implements a Pandas DataFrame Agent powered by Mistral (via Ollama) that can reason dynamically
over e-commerce operational data (products + orders).

Objective:
Provide the LLM with a structured dataset that merges product information and
delivery performance, allowing it to generate custom analytics queries (group by, mean, etc.)
on demand, without predefining every possible aggregation.
"""

import warnings

import logging
import os
import pandas as pd
from langchain.agents import AgentExecutor
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent

from app.constants import ORDERS_PATH, PRODUCTS_PATH


# Configure basic logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_errors.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data():
    """
    Load and merge product and order datasets into a single analytical DataFrame.

    Business logic:
    - Expose a structured, aggregated view of the e-commerce data for safe reasoning.
    - Compute lightweight features such as `late_rate` (percentage of late deliveries)
      to provide the LLM with stable, interpretable signals for the most common metrics.
    - This serves as minimal feature engineering: the agent can still compute
      additional aggregations dynamically, but avoids recomputing trivial statistics.
    """
    df_products = pd.read_csv(PRODUCTS_PATH)
    df_orders = pd.read_csv(ORDERS_PATH)

    late_rate = (
        df_orders.groupby("product_id")["delivered_late"]
        .mean()
        .reset_index(name="late_rate")
    )

    df = pd.merge(df_products, late_rate, on="product_id", how="left")
    return df


def get_pandas_agent():
    """Create and configure the Pandas DataFrame agent with logging and error handling."""
    # Setup LLM
    llm = Ollama(model="mistral:instruct", temperature=0)

    # Load data
    df = load_data()

    # Create base agent
    base_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Wrap with executor that logs parsing issues instead of crashing
    agent = AgentExecutor.from_agent_and_tools(
        base_agent.agent,
        base_agent.tools,
        handle_parsing_errors=lambda e: f"Handled parsing error: {str(e)}",
        verbose=True,
    )

    return agent


def ask_agent(question: str):
    """
    Query the Pandas Agent with a natural language question.

    Args:
        question (str): Example - "Which categories have the highest return rate?"

    Returns:
        str: The LLM-generated answer.
    """
    agent = get_pandas_agent()
    response = agent.run(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    return response


if __name__ == "__main__":
    ask_agent("Which product categories have the highest average return rate?")
