"""
Telegram Configuration Test Utility

This script helps you test your Telegram bot configuration.
Run this before starting the main application to verify everything is set up correctly.
"""

import os
import sys

from dotenv import load_dotenv


def print_section(title):
    """Print a section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def check_env_file():
    """Check if .env file exists"""
    print_section("1. Checking .env file")

    if os.path.exists(".env"):
        print("✅ .env file found")
        return True
    else:
        print("❌ .env file not found")
        print("\n💡 Create a .env file:")
        print("   1. Copy .env.example to .env")
        print("   2. Edit .env and add your TELEGRAM_TOKEN and CHAT_ID")
        print("\n   Example:")
        print("   TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
        print("   CHAT_ID=123456789")
        return False


def check_dotenv():
    """Check if python-dotenv is installed"""
    print_section("2. Checking python-dotenv")

    try:
        import dotenv

        print(f"✅ python-dotenv installed (version {dotenv.__version__})")
        return True
    except ImportError:
        print("❌ python-dotenv not installed")
        print("\n💡 Install it:")
        print("   pip install python-dotenv")
        return False


def check_telegram_library():
    """Check if python-telegram-bot is installed"""
    print_section("3. Checking python-telegram-bot")

    try:
        import telegram

        print(f"✅ python-telegram-bot installed (version {telegram.__version__})")
        return True
    except ImportError:
        print("❌ python-telegram-bot not installed")
        print("\n💡 Install it:")
        print("   pip install python-telegram-bot")
        return False


def check_environment_variables():
    """Check if environment variables are loaded"""
    print_section("4. Checking Environment Variables")

    load_dotenv()

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("CHAT_ID")

    issues = []

    # Check TOKEN
    if not token:
        print("❌ TELEGRAM_TOKEN not set")
        issues.append("TELEGRAM_TOKEN")
    elif token.strip() == "" or token == "your_telegram_bot_token_here":
        print("❌ TELEGRAM_TOKEN is empty or using default value")
        issues.append("TELEGRAM_TOKEN")
    else:
        # Validate token format (basic check)
        if ":" in token and len(token) > 20:
            masked_token = token[:10] + "..." + token[-5:]
            print(f"✅ TELEGRAM_TOKEN is set ({masked_token})")
        else:
            print("⚠️  TELEGRAM_TOKEN format looks invalid")
            print("   Expected format: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
            issues.append("TELEGRAM_TOKEN format")

    # Check CHAT_ID
    if not chat_id:
        print("❌ CHAT_ID not set")
        issues.append("CHAT_ID")
    elif chat_id.strip() == "" or chat_id == "your_telegram_chat_id_here":
        print("❌ CHAT_ID is empty or using default value")
        issues.append("CHAT_ID")
    else:
        # Validate chat_id format
        try:
            int(chat_id)
            print(f"✅ CHAT_ID is set ({chat_id})")
        except ValueError:
            print(f"⚠️  CHAT_ID format looks invalid: {chat_id}")
            print("   Expected format: 123456789 (numbers only)")
            issues.append("CHAT_ID format")

    return len(issues) == 0, token, chat_id


def test_bot_connection(token):
    """Test connection to Telegram API"""
    print_section("5. Testing Bot Connection")

    try:
        from telegram import Bot
        import asyncio

        async def test_connection():
            bot = Bot(token=token)
            try:
                me = await bot.get_me()
                print(f"✅ Connected to Telegram API")
                print(f"   Bot name: @{me.username}")
                print(f"   Bot ID: {me.id}")
                return True
            except Exception as e:
                print(f"❌ Failed to connect: {str(e)}")
                return False

        # Run async test
        result = asyncio.run(test_connection())
        return result

    except Exception as e:
        print(f"❌ Error testing connection: {str(e)}")
        return False


def test_send_message(token, chat_id):
    """Test sending a message"""
    print_section("6. Testing Message Send")

    try:
        from telegram import Bot
        import asyncio

        async def send_test_message():
            bot = Bot(token=token)
            try:
                message = await bot.send_message(
                    chat_id=chat_id,
                    text="🧪 Test message from Trading Bot configuration checker!\n\n"
                    "✅ Your Telegram bot is configured correctly!",
                )
                print(f"✅ Test message sent successfully!")
                print(f"   Message ID: {message.message_id}")
                return True
            except Exception as e:
                print(f"❌ Failed to send message: {str(e)}")
                if "Chat not found" in str(e):
                    print("\n💡 Possible causes:")
                    print("   1. Wrong CHAT_ID")
                    print("   2. You haven't started a chat with your bot yet")
                    print("   3. The bot is blocked by the user")
                    print("\n   Solution: Open Telegram and send /start to your bot")
                return False

        # Run async test
        result = asyncio.run(send_test_message())
        return result

    except Exception as e:
        print(f"❌ Error sending message: {str(e)}")
        return False


def print_summary(results):
    """Print test summary"""
    print_section("Test Summary")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed} / {total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your Telegram bot is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Start your application")
        print("   2. Send /start to your bot in Telegram")
        print("   3. Try /status to verify the bot is responding")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n📚 For detailed setup instructions, see:")
        print("   docs/guides/TELEGRAM_SETUP.md")


def main():
    """Run all tests"""
    print("\n🤖 Telegram Bot Configuration Test")
    print("=" * 60)

    results = {}

    # Test 1: Check .env file
    results["1. .env file exists"] = check_env_file()
    if not results["1. .env file exists"]:
        print_summary(results)
        return 1

    # Test 2: Check python-dotenv
    results["2. python-dotenv installed"] = check_dotenv()

    # Test 3: Check telegram library
    results["3. python-telegram-bot installed"] = check_telegram_library()

    # If libraries not installed, stop here
    if not all(
        [
            results["2. python-dotenv installed"],
            results["3. python-telegram-bot installed"],
        ]
    ):
        print_summary(results)
        return 1

    # Test 4: Check environment variables
    env_ok, token, chat_id = check_environment_variables()
    results["4. Environment variables valid"] = env_ok

    if not env_ok:
        print_summary(results)
        return 1

    # Test 5: Test bot connection
    results["5. Bot connection"] = test_bot_connection(token)

    if not results["5. Bot connection"]:
        print("\n💡 If the token is correct but connection fails:")
        print("   1. Check your internet connection")
        print("   2. Verify the bot hasn't been deleted in @BotFather")
        print("   3. Try generating a new token with @BotFather")
        print_summary(results)
        return 1

    # Test 6: Test message send
    print("\n⚠️  About to send a test message to your Telegram...")
    response = input("Continue? (y/n): ").strip().lower()

    if response == "y":
        results["6. Message send"] = test_send_message(token, chat_id)
    else:
        print("⏭️  Skipped message send test")
        results["6. Message send"] = None

    # Print summary
    print_summary(results)

    # Return success/failure
    return 0 if all(r for r in results.values() if r is not None) else 1


if __name__ == "__main__":
    sys.exit(main())
