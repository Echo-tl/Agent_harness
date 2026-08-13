import os
from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
response = client.search("AI agent GitHub projects 2025", max_results=3)
print(f"结果数: {len(response.get('results', []))}")
for r in response.get("results", []):
    print(f"  - {r['title'][:60]}: {r['url'][:60]}")