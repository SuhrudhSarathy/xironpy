# import asyncio
# import websockets
# import logging

# logging.basicConfig(level=logging.INFO)
# from random import choice


# async def websocket_client():
#     uri = "ws://localhost:8765"
#     async with websockets.connect(uri) as websocket:
#         logging.info(f"Connected to {uri}")

#         # Check and print custom headers
#         for header, value in websocket.response_headers.items():
#             if header.startswith("mycustom") or header.startswith("some_tungstenite"):
#                 logging.info(f"Received custom header: {header}: {value}")

#         while True:
#             # Get user input
#             message = str(choice(list(range(1, 100000))))

#             if message.lower() == "quit":
#                 break

#             # Send message
#             await websocket.send(message)
#             logging.info(f"Sent: {message}")

#             # Receive response
#             # response = await websocket.recv()
#             # logging.info(f"Received: {response}")


# async def main():
#     try:
#         await websocket_client()
#     except websockets.exceptions.ConnectionClosed:
#         logging.error("WebSocket connection closed unexpectedly")
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")


# if __name__ == "__main__":
#     asyncio.run(main())

from xiron_py.ws_comms import XironContext

if __name__ == "__main__":
    ctx = XironContext()
