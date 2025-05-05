from langchain.tools import tool
import requests

@tool
def search_papers(query: str) -> str:
    """
    Search Semantic Scholar for academic papers and return title, abstract, and URL.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,abstract,url"
    response = requests.get(url)
    results = response.json().get("data", [])

    if not results:
        return "No papers found."

    output = ""
    for paper in results:
        output += f"Title: {paper['title']}\n"
        output += f"Abstract: {paper.get('abstract', 'No abstract available.')}\n"
        output += f"URL: {paper['url']}\n\n"
    return output
