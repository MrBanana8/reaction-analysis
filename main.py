import asyncio

from dotenv import load_dotenv

from websocket_server import VideoEmotionServer


def main():
    load_dotenv()

    server = VideoEmotionServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
