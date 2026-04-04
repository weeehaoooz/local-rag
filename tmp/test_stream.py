import asyncio
from llama_index.llms.ollama import Ollama

async def main():
    llm = Ollama(model="llama3:latest", request_timeout=30.0)
    print("Testing astream_complete...")
    try:
        resp_gen = await llm.astream_complete("Hello, say 'testing' and stop.")
        async for chunk in resp_gen:
            print("CHUNK:", chunk.delta)
    except Exception as e:
        print("ERROR:", e)

asyncio.run(main())
