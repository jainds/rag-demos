import asyncio
import nest_asyncio

# Use default event loop policy
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Create and set event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Apply nest_asyncio
nest_asyncio.apply()

# Import app after event loop setup
from app.main import app

# This is needed for ASGI servers
application = app 