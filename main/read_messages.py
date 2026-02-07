import discord
from datetime import datetime

# -------------------------------
# SET THESE VALUES
# -------------------------------
MY_USER_ID = 537318854708428820       # your Discord user ID (int)
MY_CHANNEL_ID = 1467660677538775296   # VC text chat channel ID
BOT_TOKEN = ""
OUTPUT_FILE = "discord_export.txt"
# -------------------------------

# Enable intents
intents = discord.Intents.default()
intents.message_content = True

# Create client
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}!')

@client.event
async def on_message(message):
    # Only the target channel
    if message.channel.id != MY_CHANNEL_ID:
        return

    # Ignore bot itself
    if message.author == client.user:
        return

    # Only YOUR messages
    if message.author.id != MY_USER_ID:
        return

    # Timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Line to export
    line = f"[{timestamp}] {message.content}\n"

    # Write to file (append)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(line)

    # Also print to terminal
    print(line.strip())

    # Example command logic
    if message.content.startswith('$hello'):
        print('hello')

# Run the bot
client.run(BOT_TOKEN)
