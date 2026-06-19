import os
import json
import urllib.parse
import requests
from bs4 import BeautifulSoup  # Inherited from system packages if available

class WebSearch:
    def __init__(self, google_api_key=None, google_cx=None, searxng_url=None):
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.google_cx = google_cx or os.environ.get("GOOGLE_CX")
        # Default to a reliable public instance if not specified
        self.searxng_url = searxng_url or os.environ.get("SEARXNG_URL", "https://searx.be")
        # Persistent session for TCP connection pooling (reuses SSL handshakes)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"})

    def search(self, query, max_results=5):
        """
        Search the web. Priority: Google API -> SearXNG -> DuckDuckGo.
        """
        if self.google_api_key and self.google_cx:
            print(f"Searching Google for: '{query}'")
            return self._google_search(query, max_results)
        elif self.searxng_url:
            print(f"Searching SearXNG ({self.searxng_url}) for: '{query}'")
            return self._searxng_search(query, max_results)
        else:
            print(f"Searching DuckDuckGo (free fallback) for: '{query}'")
            return self._ddg_search_api(query, max_results)

    def _searxng_search(self, query, max_results=5):
        """Search using SearXNG JSON API."""
        try:
            safe_query = urllib.parse.quote(query)
            url = f"{self.searxng_url.rstrip('/')}/search?q={safe_query}&format=json"
            
            response = self._session.get(url, timeout=5.0)
            data = response.json()
                
            results = []
            if "results" in data:
                for item in data["results"][:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("content", "")
                    })
                    
            if results:
                return results
            else:
                print("SearXNG returned 0 results. Falling back to DuckDuckGo...")
                return self._ddg_search_api(query, max_results)
        except Exception as e:
            print(f"SearXNG search failed: {str(e)}. Falling back to DuckDuckGo...")
            return self._ddg_search_api(query, max_results)

    def _google_search(self, query, max_results=5):
        """Search using Google Custom Search JSON API."""
        try:
            safe_query = urllib.parse.quote(query)
            url = f"https://www.googleapis.com/customsearch/v1?key={self.google_api_key}&cx={self.google_cx}&q={safe_query}&num={max_results}"
            
            response = self._session.get(url, timeout=5)
            data = response.json()
                
            results = []
            if "items" in data:
                for item in data["items"]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            return results
        except Exception as e:
            print(f"Google search API failed: {str(e)}. Falling back to DuckDuckGo...")
            return self._ddg_search_api(query, max_results)

    def _ddg_search_api(self, query, max_results=5):
        """
        Search using duckduckgo_search library if available,
        or fall back to a light web request scraper if not.
        """
        try:
            import warnings
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
                if results:
                    return results
                else:
                    print("DuckDuckGo search library returned 0 results. Trying HTML scraper...")
                    return self._ddg_html_scraper(query, max_results)
        except ImportError:
            # Fallback to direct HTML search scraping or HTML API
            return self._ddg_html_scraper(query, max_results)
        except Exception as e:
            print(f"DuckDuckGo search failed: {str(e)}")
            return self._ddg_html_scraper(query, max_results)

    def _ddg_html_scraper(self, query, max_results=5):
        """Scrape DuckDuckGo HTML search page as a robust fallback without library dependencies."""
        try:
            # DuckDuckGo HTML version is lightweight and scrapeable
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            
            response = self._session.get(url, timeout=5.0)
            html = response.content
                
            soup = BeautifulSoup(html, "html.parser")
            results = []
            
            # Find search result divs on ddg html page
            for result_div in soup.find_all("div", class_="result"):
                if len(results) >= max_results:
                    break
                    
                title_elem = result_div.find("a", class_="result__a")
                snippet_elem = result_div.find("a", class_="result__snippet")
                
                if title_elem:
                    title = title_elem.text.strip()
                    link = title_elem.get("href", "")
                    
                    # Handle redirect link mapping
                    if "uddg=" in link or link.startswith("//"):
                        if link.startswith("//"):
                            full_url = "https:" + link
                        else:
                            full_url = link
                        parsed = urllib.parse.urlparse(full_url)
                        qs = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in qs:
                            link = qs["uddg"][0]
                            
                    snippet = snippet_elem.text.strip() if snippet_elem else ""
                    
                    if link:
                        results.append({
                            "title": title,
                            "link": link,
                            "snippet": snippet
                        })
            return results
        except Exception as e:
            print(f"HTML scraper fallback failed: {str(e)}")
            return []

    def scrape_url(self, url):
        """Deep scrape the full text of a webpage."""
        try:
            print(f"Deep scraping: {url}")
            response = self._session.get(url, timeout=8.0)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove junk elements
            for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
                el.decompose()
                
            text = soup.get_text(separator="\n")
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)
            
            # Cap at 15000 characters to prevent context overflow
            return cleaned_text[:15000]
        except Exception as e:
            print(f"Failed to deep scrape {url}: {str(e)}")
            return ""

if __name__ == "__main__":
    # Test search
    ws = WebSearch()
    res = ws.search("Intel Core i5-1235u Xe graphics specification", 3)
    print(json.dumps(res, indent=2))

