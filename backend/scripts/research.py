import os
import sys
import argparse
from typing import List, Dict

# Add backend directory to sys.path for internal imports
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from research.planner import ResearchPlanner
from research.searcher import ResearchSearcher
from research.downloader import ResearchDownloader
from research.web_searcher import WebSearcher
from research.scraper import WebScraper
from config import DATA_DIR

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt
from rich import print as rprint

class ResearchCLI:
    def __init__(self):
        self.console = Console()
        self.planner = ResearchPlanner()
        self.arxiv_searcher = ResearchSearcher(max_results_per_query=5)
        self.web_searcher = WebSearcher(max_results=5)
        
        self.pdf_downloader = ResearchDownloader(data_dir=os.path.join(DATA_DIR, "research"))
        self.web_scraper = WebScraper(data_dir=os.path.join(DATA_DIR, "web_research"))
        
        self.mode = "arxiv" # arxiv, web, news
        self.current_topic = None
        self.results = []

    def display_welcome(self):
        self.console.print(Panel.fit(
            "[bold cyan]Research Engine CLI[/bold cyan]\n"
            f"[italic]Powered by Ollama | Current Mode: [bold yellow]{self.mode.upper()}[/bold yellow][/italic]\n\n"
            "Type a topic to research, or use [bold green]/help[/bold green] for commands.",
            title="Welcome", border_style="blue"
        ))

    def help(self):
        table = Table(title="Available Commands", border_style="cyan")
        table.add_column("Command", style="bold green")
        table.add_column("Description")
        table.add_row("/topic <topic>", "Start research on a specific topic.")
        table.add_row("/mode <mode>", "Switch between [yellow]arxiv[/], [yellow]web[/], [yellow]news[/], and [yellow]deep[/].")
        table.add_row("/limit <n>", "Set max results per query (default: 5).")
        table.add_row("/clear", "Clear results and history.")
        table.add_row("/exit", "Exit the Research Engine.")
        table.add_row("<any text>", "Treated as a research topic in current mode.")
        self.console.print(table)

    def run_research(self, topic: str):
        self.current_topic = topic
        
        # 1. Planning phase
        with self.console.status(f"[bold yellow]Analyzing topic: {topic} ({self.mode}) via Ollama...[/]", spinner="dots"):
            plan = self.planner.generate_plan(topic, mode=self.mode)
        
        self.console.print(f"\n[bold green]Research Objective:[/] {plan['objective']}")
        self.console.print(f"[bold cyan]Planned Queries:[/] {', '.join(plan['queries'])}\n")
        
        # 2. Searching phase
        with self.console.status(f"[bold yellow]Searching {self.mode.upper()}...[/]", spinner="earth"):
            if self.mode == "arxiv":
                self.results = self.arxiv_searcher.search(plan['queries'])
            elif self.mode == "web":
                self.results = self.web_searcher.search_text(plan['queries'])
            elif self.mode == "news":
                self.results = self.web_searcher.search_news(plan['queries'])
            elif self.mode == "deep":
                # Deep mode combines ArXiv and Web
                arxiv_queries = [q for q in plan['queries'] if any(x in q.lower() for x in ['arxiv', 'paper', 'journal'])]
                web_queries = [q for q in plan['queries'] if q not in arxiv_queries]
                
                # If planner didn't distinguish, just split them
                if not arxiv_queries:
                    arxiv_queries = plan['queries'][:len(plan['queries'])//2]
                    web_queries = plan['queries'][len(plan['queries'])//2:]
                
                self.results = self.arxiv_searcher.search(arxiv_queries)
                self.results += self.web_searcher.search_text(web_queries)

        if not self.results:
            self.console.print(f"[bold red]No results found in {self.mode} mode.[/]")
            return

        # 2.5 Terminology Discovery (Deep or Web mode)
        if self.mode in ["deep", "web", "arxiv"]:
            with self.console.status("[bold magenta]Discovering key terminologies...[/]", spinner="simpleDots"):
                terms = self.planner.discover_terms(self.results)
                if terms:
                    self.console.print(f"[bold magenta]Found technical terms:[/] {', '.join(terms)}")
                    definitions = self.web_searcher.search_definitions(terms)
                    # Add definitions to the top of results
                    self.results = definitions + self.results

        # 3. Display Results
        table = Table(title=f"{self.mode.upper()} Results for: {topic}", show_lines=True)
        table.add_column("Idx", justify="center", style="dim")
        table.add_column("Source", style="bold cyan")
        table.add_column("Title / Authors", style="bold white")
        table.add_column("Summary / Snippet", style="dim")
        
        for i, r in enumerate(self.results, 1):
            if r.get('result_obj'): # ArXiv
                source = "ArXiv"
                title_line = f"{r['title']}\n[italic]{', '.join(r['authors'][:2])}[/]"
                content = r.get('summary', '')[:200] + "..."
            else:
                source = r.get('source', 'web').capitalize()
                title_line = r['title']
                content = r.get('snippet', '')[:200] + "..."
            
            table.add_row(str(i), source, title_line, content)
        
        self.console.print(table)
        
        # 4. Save/Download phase
        has_arxiv = any(r.get('result_obj') for r in self.results)
        has_web = any(r.get('source') in ['web', 'news', 'dictionary'] for r in self.results)
        
        actions = []
        if has_arxiv: actions.append("Download PDFs")
        if has_web: actions.append("Scrape full articles/definitions to TXT")
        
        action_str = " & ".join(actions)
        confirm = Prompt.ask(f"\n[bold]{action_str} for these results?[/]", choices=["y", "n"], default="y")
        
        if confirm.lower() == 'y':
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            ) as progress:
                label = "Downloading" if self.mode == "arxiv" else "Scraping"
                overall_task = progress.add_task(f"[cyan]{label} data...", total=len(self.results))
                
                for r in self.results:
                    if r.get('result_obj'): # ArXiv paper
                        path = self.pdf_downloader.download(r)
                    else: # Web or Dictionary
                        path = self.web_scraper.scrape_to_file(r)
                        
                    if path:
                        progress.print(f"✅ [green]Saved:[/] {os.path.basename(path)}")
                    else:
                        progress.print(f"❌ [red]Failed:[/] {r['title'][:50]}...")
                    progress.advance(overall_task)
            
            self.console.print(f"\n[bold green]Success![/] Data available in research and web_research directories.")

    def start(self):
        self.display_welcome()
        while True:
            try:
                user_input = Prompt.ask(f"\n[bold cyan]research[/bold cyan] [dim]({self.mode})[/] [white]❯[/white]").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                    self.console.print("[bold yellow]Goodbye![/]")
                    break
                
                if user_input.startswith("/"):
                    parts = user_input.split(" ", 1)
                    cmd = parts[0].lower()
                    
                    if cmd == "/help":
                        self.help()
                    elif cmd == "/mode":
                        if len(parts) > 1:
                            new_mode = parts[1].lower()
                            if new_mode in ["arxiv", "web", "news", "deep"]:
                                self.mode = new_mode
                                self.console.print(f"[green]Switched to [bold]{new_mode.upper()}[/] mode.[/]")
                            else:
                                self.console.print("[red]Invalid mode. Choose: arxiv, web, news, or deep.[/]")
                        else:
                            self.console.print(f"[cyan]Current mode: {self.mode}. Usage: /mode <arxiv|web|news|deep>[/]")
                    elif cmd == "/topic":
                        if len(parts) > 1:
                            self.run_research(parts[1])
                        else:
                            self.console.print("[red]Please specify a topic. Usage: /topic <topic>[/]")
                    elif cmd == "/limit":
                        if len(parts) > 1 and parts[1].isdigit():
                            limit = int(parts[1])
                            self.arxiv_searcher.max_results = limit
                            self.web_searcher.max_results = limit
                            self.console.print(f"[green]Limit set to {limit}[/]")
                        else:
                            self.console.print("[red]Invalid limit. Usage: /limit <number>[/]")
                    elif cmd == "/clear":
                        self.results = []
                        self.current_topic = None
                        self.console.clear()
                        self.display_welcome()
                    else:
                        self.console.print(f"[red]Unknown command: {cmd}. Type /help for assistance.[/]")
                else:
                    self.run_research(user_input)
                    
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Exiting...[/]")
                break
            except Exception as e:
                self.console.print(f"\n[bold red]Error:[/] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Research Engine CLI")
    parser.add_argument("--topic", type=str, help="Research topic to run immediately")
    parser.add_argument("--mode", type=str, default="arxiv", help="Initial mode (arxiv, web, news, deep)")
    args = parser.parse_args()

    cli = ResearchCLI()
    cli.mode = args.mode
    if args.topic:
        cli.run_research(args.topic)
    else:
        cli.start()

if __name__ == "__main__":
    main()
