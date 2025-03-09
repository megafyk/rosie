import asyncio
import os
from typing import Dict, List

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Import LangGraph components
from chat import graph
from search import search

# Load environment variables
load_dotenv()

# Get Telegram token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Store conversation history by user ID
user_messages: Dict[int, List[Dict[str, str]]] = {}
# Store thread IDs for LangGraph sessions
user_threads: Dict[int, str] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message when /start is issued."""
    await update.message.reply_text(
        "Hi! I'm your RAG-powered assistant. Ask me anything!"
    )
    user_id = update.effective_user.id
    user_messages[user_id] = []
    user_threads[user_id] = str(user_id)  # Use user_id as thread_id


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message when /help is issued."""
    help_text = (
        "Here's how to use me:\n"
        "- Ask me any question\n"
        "- I'll search my knowledge base and respond\n"
        "- Use /reset to clear our conversation\n"
        "- Use /help to see this message"
    )
    await update.message.reply_text(help_text)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset conversation history."""
    user_id = update.effective_user.id
    if user_id in user_messages:
        user_messages[user_id] = []
        # Reset thread by using a new thread_id
        user_threads[user_id] = f"{user_id}-{len(user_messages)}"
    await update.message.reply_text("Conversation history cleared!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user messages with LangGraph."""
    user_id = update.effective_user.id
    query = update.message.text

    # Initialize if needed
    if user_id not in user_messages:
        user_messages[user_id] = []
        user_threads[user_id] = str(user_id)

    # Get thread ID for this user
    thread_id = user_threads.get(user_id, str(user_id))

    # Add user message to history
    user_messages[user_id].append({"role": "user", "content": query})

    # Send "typing" indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Get RAG context to enhance the query
    rag_context = search(query)
    enhanced_query = f"{query}\n\nContext: {rag_context}"

    # Send initial message for streaming updates
    response_message = await update.message.reply_text("Generating response...")

    try:
        # Set up config for LangGraph
        config = {"configurable": {"thread_id": thread_id}}

        # Prepare full response
        full_response = ""
        buffer = ""
        last_update_time = asyncio.get_event_loop().time()

        # Stream response from LangGraph
        async for event in graph.astream(
                {"messages": [{"role": "user", "content": enhanced_query}]},
                config,
                stream_mode="values"
        ):
            for value in event.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    # Extract assistant message from the response
                    for msg in value:
                        if msg.get("role") == "assistant" and msg.get("content"):
                            content = msg["content"]
                            # Only process new content
                            if content and content not in full_response:
                                new_content = content[len(full_response):]
                                full_response = content
                                buffer += new_content

                                # Update message periodically to avoid rate limiting
                                current_time = asyncio.get_event_loop().time()
                                if len(buffer) >= 20 or (current_time - last_update_time) > 1.0:
                                    await response_message.edit_text(full_response)
                                    buffer = ""
                                    last_update_time = current_time
                                    await asyncio.sleep(0.1)

        # Final update with complete response
        if full_response:
            await response_message.edit_text(full_response)
            # Add AI response to history
            user_messages[user_id].append({"role": "assistant", "content": full_response})
        else:
            await response_message.edit_text("I couldn't generate a response. Please try again.")

    except Exception as e:
        await response_message.edit_text(f"Error: {str(e)}\n\nPlease try again.")


def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        return

    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    print("Starting bot...")
    application.run_polling()


if __name__ == "__main__":
    main()
