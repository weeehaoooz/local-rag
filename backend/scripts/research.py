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

        self.mode = "arxiv"  # arxiv, web, news, deep
        self.current_topic = None
        self.results: List[Dict] = []
        self.plan: Dict = {}
        self.synthesis: str = ""

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def display_welcome(self):
        self.console.print(Panel.fit(
            "[bold cyan]Advanced Research Engine[/bold cyan]\n"
            f"[italic]Powered by Ollama | Current Mode: [bold yellow]{self.mode.upper()}[/bold yellow][/italic]\n\n"
            "Type a topic to research, or use [bold green]/help[/bold green] for interactive commands.",
            title="Welcome", border_style="blue"
        ))

    def help(self):
        table = Table(title="Available Commands", border_style="cyan")
        table.add_column("Command", style="bold green")
        table.add_column("Description")
        table.add_row("/topic <topic>", "Start research on a specific topic.")
        table.add_row("/mode <mode>", "Switch between [yellow]arxiv, web, news, deep[/].")
        table.add_row("/ask <query>", "Ask a question about the current research findings.")
        table.add_row("/analyze <idx>", "Get a deep-dive analysis of a specific result.")
        table.add_row("/refine <feedback>", "Refine the current research plan.")
        table.add_row("/synthesize", "Regenerate the research synthesis report.")
        table.add_row("/limit <n>", "Set max results per query (default: 5).")
        table.add_row("/clear", "Clear results and history.")
        table.add_row("/exit", "Exit the Research Engine.")
        self.console.print(table)

    # ------------------------------------------------------------------
    # Core research logic
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(results: List[Dict]) -> List[Dict]:
        seen: set = set()
        unique: List[Dict] = []
        for r in results:
            key = r.get("id") or r.get("link")
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def run_research(self, topic: str):
        self.current_topic = topic
        self.results = []
        self.synthesis = ""

        # 1. Planning phase
        with self.console.status(f"[bold yellow]Planning research for: {topic}...[/]", spinner="dots"):
            self.plan = self.planner.generate_plan(topic, mode=self.mode)

        self.interactive_planning()

    def interactive_planning(self):
        while True:
            self.console.print(Panel(
                f"[bold green]Objective:[/] {self.plan['objective']}\n"
                f"[bold cyan]Queries:[/] {', '.join(self._get_query_labels(self.plan['queries']))}",
                title="Proposed Research Plan", border_style="green"
            ))

            choice = Prompt.ask(
                "\n[bold]Execute plan, [yellow]refine[/] it, or [red]cancel[/]?[/]",
                choices=["e", "r", "c"], default="e"
            )

            if choice == "e":
                self.execute_search()
                break
            elif choice == "r":
                feedback = Prompt.ask("[yellow]What should I change in the plan?[/]")
                with self.console.status("[bold yellow]Refining plan...[/]", spinner="dots"):
                    self.plan = self.planner.refine_plan(self.current_topic, self.plan, feedback)
            else:
                self.console.print("[red]Research cancelled.[/]")
                break

    def _get_query_labels(self, queries: List) -> List[str]:
        labels = []
        for q in queries:
            if isinstance(q, dict):
                labels.append(f"{q['query']} [dim]({q.get('backend', '?')})[/]")
            else:
                labels.append(str(q))
        return labels

    def execute_search(self):
        # 2. Searching phase
        with self.console.status(f"[bold yellow]Searching {self.mode.upper()}...[/]", spinner="earth"):
            if self.mode == "arxiv":
                self.results = self.arxiv_searcher.search(self.plan["queries"])
            elif self.mode == "web":
                self.results = self.web_searcher.search_text(self.plan["queries"])
            elif self.mode == "news":
                self.results = self.web_searcher.search_news(self.plan["queries"])
            elif self.mode == "deep":
                arxiv_queries = [q["query"] for q in self.plan["queries"] if isinstance(q, dict) and q.get("backend") == "arxiv"]
                web_queries = [q["query"] for q in self.plan["queries"] if isinstance(q, dict) and q.get("backend") == "web"]
                
                arxiv_res = self.arxiv_searcher.search(arxiv_queries) if arxiv_queries else []
                web_res = self.web_searcher.search_text(web_queries) if web_queries else []
                self.results = self._deduplicate(arxiv_res + web_res)

        if not self.results:
            self.console.print("[bold red]No results found.[/]")
            return

        # 2.5 Terminology Discovery
        if self.mode in ["deep", "web", "news"]:
            with self.console.status("[bold magenta]Discovering technical terms...[/]", spinner="simpleDots"):
                terms = self.planner.discover_terms(self.results)
                if terms:
                    self.console.print(f"[bold magenta]Key Terms:[/] {', '.join(terms)}")
                    definitions = self.web_searcher.search_definitions(terms)
                    self.results = self._deduplicate(definitions + self.results)

        # 3. Synthesis Phase
        with self.console.status("[bold blue]Synthesizing Research Report...[/]", spinner="aesthetic"):
            self.synthesis = self.planner.synthesize_results(self.current_topic, self.results)

        self.display_results()

    def display_results(self):
        # Display Synthesis
        self.console.print(Panel(
            self.synthesis, title=f"Research Report: {self.current_topic}", border_style="blue"
        ))

        # Display Results Table
        table = Table(title="Retrieved Sources", show_lines=True)
        table.add_column("Idx", justify="center", style="dim")
        table.add_column("Source", style="bold cyan")
        table.add_column("Title / Snippet", style="white")

        for i, r in enumerate(self.results, 1):
            source = r.get("source", "web").capitalize()
            title = r.get("title", "(no title)")
            content = r.get("snippet", r.get("summary", ""))[:150] + "..."
            table.add_row(str(i), source, f"[bold]{title}[/]\n[dim]{content}[/]")

        self.console.print(table)
        self.console.print(
            "\n[dim]Use [bold]/ask[/] to query these results, [bold]/analyze <idx>[/] for deep-dives, or [bold]/exit[/] to finish.[/]"
        )

    def chat_with_results(self, question: str):
        if not self.results:
            self.console.print("[red]No research context available. Start a topic first.[/]")
            return
        
        with self.console.status("[bold cyan]Consulting findings...[/]", spinner="bouncingBall"):
            answer = self.planner.chat_with_results(self.current_topic, question, self.results)
        
        self.console.print(Panel(answer, title="Research Assistant", border_style="cyan"))

    def analyze_item(self, idx_str: str):
        try:
            idx = int(idx_str) - 1
            if 0 <= idx < len(self.results):
                result = self.results[idx]
                with self.console.status(f"[bold magenta]Analyzing result {idx+1}...[/]", spinner="pong"):
                    analysis = self.planner.analyze_result(self.current_topic, result)
                self.console.print(Panel(analysis, title=f"Deep Analysis: {result['title']}", border_style="magenta"))
            else:
                self.console.print(f"[red]Invalid index: {idx_str}[/]")
        except ValueError:
            self.console.print("[red]Please provide a numeric index.[/]")

    # ------------------------------------------------------------------
    # Interactive CLI loop
    # ------------------------------------------------------------------

    def start(self):
        self.display_welcome()
        while True:
            try:
                user_input = Prompt.ask(
                    f"\n[bold cyan]research[/bold cyan] [dim]({self.mode})[/] [white]❯[/white]"
                ).strip()

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
                                self.console.print("[red]Invalid mode.[/]")
                    elif cmd == "/topic":
                        if len(parts) > 1:
                            self.run_research(parts[1])
                        else:
                            self.console.print("[red]Usage: /topic <topic>[/]")
                    elif cmd == "/ask":
                        if len(parts) > 1:
                            self.chat_with_results(parts[1])
                        else:
                            self.console.print("[red]Usage: /ask <question>[/]")
                    elif cmd == "/analyze":
                        if len(parts) > 1:
                            self.analyze_item(parts[1])
                        else:
                            self.console.print("[red]Usage: /analyze <index>[/]")
                    elif cmd == "/refine":
                        if self.current_topic and self.plan:
                            feedback = parts[1] if len(parts) > 1 else Prompt.ask("[yellow]Refinement feedback?[/]")
                            with self.console.status("[bold yellow]Refining...[/]"):
                                self.plan = self.planner.refine_plan(self.current_topic, self.plan, feedback)
                            self.interactive_planning()
                        else:
                            self.console.print("[red]No active research to refine.[/]")
                    elif cmd == "/synthesize":
                        if self.results:
                            with self.console.status("[bold blue]Regenerating report...[/]"):
                                self.synthesis = self.planner.synthesize_results(self.current_topic, self.results)
                            self.display_results()
                        else:
                            self.console.print("[red]No results to synthesize.[/]")
                    elif cmd == "/clear":
                        self.results = []
                        self.current_topic = None
                        self.console.clear()
                        self.display_welcome()
                    else:
                        self.console.print(f"[red]Unknown command: {cmd}[/]")
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
