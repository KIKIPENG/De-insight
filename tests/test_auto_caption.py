"""快速測試 OpenRouter vision API（零依賴，只用 httpx）。"""
import asyncio
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


async def main():
    from settings import load_env
    env = load_env()

    model = "google/gemini-2.5-flash"
    api_key = env.get("RAG_API_KEY", "") or env.get("OPENROUTER_API_KEY", "")
    api_base = "https://openrouter.ai/api/v1"

    # 找一張測試圖
    img_path = None
    data_dir = Path.home() / ".deinsight" / "v0.6" / "projects"
    if data_dir.exists():
        for p in data_dir.rglob("*.jpg"):
            img_path = p
            break
    if not img_path:
        print("No image found")
        return

    print(f"image: {img_path.name} ({img_path.stat().st_size} bytes)")
    b64 = base64.b64encode(img_path.read_bytes()).decode()

    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                "max_tokens": 200,
            },
        )
        print(f"status: {resp.status_code}")
        data = resp.json()
        if resp.status_code == 200:
            print(f"SUCCESS: {data['choices'][0]['message']['content']}")
        else:
            print(f"ERROR: {json.dumps(data, indent=2)[:500]}")


if __name__ == "__main__":
    asyncio.run(main())
