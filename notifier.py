from telegram import Bot
from config import TELEGRAM_TOKEN, CHAT_ID

bot = Bot(token=TELEGRAM_TOKEN)

def send_message(text):
    bot.send_message(chat_id=CHAT_ID, text=text)
