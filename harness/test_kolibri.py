import asyncio
import httpx
from bench import EngineTarget, Workload, RequestResult, stream_request

async def main():
    target = EngineTarget(base_url="http://10.100.224.1:8128/v1", model="kolibri-1", extra_body={})
    workload = Workload(name="chat", prompt_tokens=1024, output_tokens=256)
    async with httpx.AsyncClient() as client:
        result = await stream_request(client, target, workload, nonce="test")
        print(result)

asyncio.run(main())