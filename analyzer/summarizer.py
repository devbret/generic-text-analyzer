import logging
import os

import anthropic

from .config import CLAUDE_MODEL

logger = logging.getLogger(__name__)


def summarize_txt_file_with_claude(filepath: str) -> str | None:
    logger.info("Reading and summarizing: %s", filepath)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        logger.error("Error reading file %s: %s", filepath, e)
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Anthropic API key not found. Please check your .env file.")
        return None

    client = anthropic.Anthropic(api_key=api_key, max_retries=5)

    try:
        logger.info("Connecting to Anthropic API for text summary...")
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4300,
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Please summarize the following document in clear and concise language suitable for a general audience.

                    Highlight:
                    - The main topics covered
                    - Any key arguments or insights
                    - Notable conclusions, themes, or recommendations

                    Here is the text:
                    {file_content}
                    """
                }
            ],
        )
        logger.info("Summary successfully generated.")
        return response.content[0].text
    except Exception as e:
        logger.error("Anthropic API error: %s", e)
        logger.error("Please check your .env file and network connection.")
        return None


def save_summary(summary: str, report_path: str) -> None:
    summary_filename = os.path.splitext(report_path)[0] + "_claude_summary.md"
    try:
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info("Claude summary saved to: %s", summary_filename)
    except Exception as e:
        logger.error("Could not save Claude summary: %s", e)
