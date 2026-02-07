"""
Discord Mosaic TTS — Message Reader
=====================================
A lightweight Discord bot that monitors a specific channel for your messages
and appends them to a log file for the main TTS engine to pick up.

Usage:
    python read_messages.py
"""

import os
import discord
from datetime import datetime

import config


# ──────────────────────────────────────────────
#  Bot setup
# ──────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

client = discord.Client(intents=intents)


def log(icon, msg):
    """Timestamped console log."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {icon} {msg}")


@client.event
async def on_ready():
    log("✔", f"Logged in as {client.user} (ID: {client.user.id})")
    log("📡", f"Target channel ID: {config.MY_CHANNEL_ID}")
    log("👤", f"Target user ID:    {config.MY_USER_ID}")
    log("📁", f"Log file path:     {config.DISCORD_LOG_FILE}")

    # Verify log file is writable
    try:
        log_dir = os.path.dirname(config.DISCORD_LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            log("📂", f"Created directory: {log_dir}")

        with open(config.DISCORD_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("")
        log("✔", f"Log file is writable")
    except Exception as e:
        log("❌", f"CANNOT WRITE TO LOG FILE: {e}")

    # Check if we can see the target channel
    channel = client.get_channel(config.MY_CHANNEL_ID)
    if channel:
        log("✔", f"Found channel: #{channel.name} in {channel.guild.name}")
    else:
        log("⚠", f"Cannot find channel {config.MY_CHANNEL_ID}!")
        log("📋", f"Servers and channels the bot can see:")
        for guild in client.guilds:
            log("  ", f"  Server: {guild.name} (ID: {guild.id})")
            for ch in guild.text_channels[:15]:
                marker = " ← MATCH" if ch.id == config.MY_CHANNEL_ID else ""
                log("  ", f"    #{ch.name} (ID: {ch.id}){marker}")

    log("🎧", "Listening for messages... (Ctrl+C to stop)")
    log("  ", "")
    log("  ", "If you type a message and NOTHING appears below,")
    log("  ", "Message Content Intent is OFF in the Developer Portal!")
    log("  ", "→ https://discord.com/developers/applications")
    log("  ", "→ Bot tab → Privileged Gateway Intents → Message Content Intent → ON")
    log("  ", "")


@client.event
async def on_message(message):
    # ════════════════════════════════════════════
    # RAW DEBUG — prints BEFORE any filtering
    # If you send a message and see NOTHING here,
    # the bot is not receiving events at all.
    # ════════════════════════════════════════════
    content_preview = message.content[:80] if message.content else "(EMPTY)"
    log("📩", f"RAW EVENT | channel={message.channel.id} | "
               f"author={message.author} ({message.author.id}) | "
               f"content='{content_preview}' | len={len(message.content)}")

    # ── Check: Message Content Intent issue ──
    if len(message.content) == 0 and not message.attachments and not message.embeds:
        log("❌", "  MESSAGE CONTENT IS EMPTY!")
        log("❌", "  This means Message Content Intent is OFF in the Developer Portal.")
        log("❌", "  Fix: https://discord.com/developers/applications → Bot → Intents → ON")
        return

    # ── Filter: wrong channel ──
    if message.channel.id != config.MY_CHANNEL_ID:
        log("  ", f"  Skipped: wrong channel (got {message.channel.id}, want {config.MY_CHANNEL_ID})")
        return

    # ── Filter: bot's own messages ──
    if message.author == client.user:
        log("🤖", f"  Skipped: bot's own message")
        return

    # ── Filter: wrong user ──
    if message.author.id != config.MY_USER_ID:
        log("👥", f"  Skipped: wrong user (got {message.author.id}, want {config.MY_USER_ID})")
        return

    # ── Filter: empty content ──
    if not message.content.strip():
        log("📭", f"  Skipped: empty message (embed/attachment only)")
        return

    # ── Write to log file ──
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message.content}\n"

    try:
        with open(config.DISCORD_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
        file_size = os.path.getsize(config.DISCORD_LOG_FILE)
        log("✅", f"  LOGGED: '{message.content}' → discord_export.txt ({file_size} bytes)")
    except Exception as e:
        log("❌", f"  FAILED TO WRITE: {e}")


@client.event
async def on_disconnect():
    log("⚠", "Bot disconnected from Discord!")


@client.event
async def on_resumed():
    log("✔", "Bot reconnected to Discord")


# ──────────────────────────────────────────────
#  Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Discord Mosaic TTS — Message Reader")
    print("=" * 55)
    print()

    log("🔧", "Checking configuration...")

    if not config.BOT_TOKEN:
        log("❌", "BOT_TOKEN is empty in config.py!")
        raise SystemExit(1)
    log("✔", f"Bot token: ...{config.BOT_TOKEN[-8:]}")

    log("📡", f"Channel ID: {config.MY_CHANNEL_ID}")
    log("👤", f"User ID:    {config.MY_USER_ID}")
    log("📁", f"Log file:   {config.DISCORD_LOG_FILE}")

    log_dir = os.path.dirname(config.DISCORD_LOG_FILE)
    if log_dir and os.path.exists(log_dir):
        log("✔", f"Log directory exists")
    else:
        log("⚠", f"Log directory missing: {log_dir}")

    print()
    log("🚀", "Connecting to Discord...")
    print()

    try:
        client.run(config.BOT_TOKEN, log_handler=None)
    except discord.LoginFailure:
        log("❌", "LOGIN FAILED — bot token is invalid or expired!")
        raise SystemExit(1)
    except Exception as e:
        log("❌", f"Fatal error: {e}")
        raise SystemExit(1)
