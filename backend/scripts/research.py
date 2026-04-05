import os
import sys
import argparse
import time
from typing import List, Dict

try:
    import readline
except ImportError:
    pass

# Add backend directory to sys.path for internal imports
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Python 3.14 / sniffio compatibility ───────────────────────────────
import sniffio_compat
sniffio_compat.apply()
# ──────────────────────────────────────────────────────────────────────

from research.planner import ResearchPlanner
from research.searcher import ResearchSearcher
from research.downloader import ResearchDownloader
from research.web_searcher import WebSearcher
from research.scraper import WebScraper
from config import DATA_DIR

# Optional import for HybridEngine (Local RAG)
try:
    from retrieval.engine import HybridEngine
    HAS_HYBRID_ENGINE = True
except ImportError:
    HAS_HYBRID_ENGINE = False

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

        self.mode = "deep"  # deep, arxiv, web, wiki, news, local
        self.engine = None  # Lazily loaded HybridEngine
        self.current_topic = None
        self.results: List[Dict] = []
        self.retained_results: List[Dict] = []
        self.plan: Dict = {}
        self.synthesis: str = ""
        self.depth = 1  # Track tree depth
        self.safe_root_topic: str = ""

    def _ask(self, text: str, choices: List[str] = None, default: str = None) -> str:
        """Readline-safe prompt wrapper to avoid backspace deleting the prompt text."""
        while True:
            opts = f" [dim]\\[{'/'.join(choices)}][/]" if choices else ""
            def_str = f" ({default})" if default else ""
            self.console.print(f"{text}{opts}{def_str}")
            
            try:
                ans = input("❯ ").strip()
            except EOFError:
                ans = ""
                
            if not ans and default is not None:
                ans = default
                
            if choices and ans not in choices:
                self.console.print(f"[red]Please enter one of: {', '.join(choices)}[/]")
                continue
                
            return ans

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
        table.add_row("/tree <topic> [depth]", "Recursive tree research (default depth=1).")
        table.add_row("/mode <mode>", "Switch between [yellow]deep, arxiv, web, wiki, news, local[/].")
        table.add_row("/ask <query>", "Ask a question about the current research findings.")
        table.add_row("/analyze <idx>", "Get a deep-dive analysis of a specific result.")
        table.add_row("/refine <feedback>", "Refine the current research plan.")
        table.add_row("/synthesize", "Regenerate the research synthesis report.")
        table.add_row("/select", "Toggle retaining current sources in memory.")
        table.add_row("/save", "Save current research results for KG indexing.")
        table.add_row("/limit <n>", "Set max results per query (default: 5).")
        table.add_row("/clear", "Clear results and history.")
        table.add_row("/exit", "Exit the Research Engine.")
        self.console.print(table)

    # ------------------------------------------------------------------
    # Core research logic
    # ------------------------------------------------------------------

    def get_combined_results(self) -> List[Dict]:
        combined = list(self.retained_results)
        retained_links = {r.get("link") for r in combined if r.get("link")}
        retained_ids = {r.get("id") for r in combined if r.get("id")}
        for r in self.results:
            if r.get("link") and r.get("link") in retained_links:
                continue
            if r.get("id") and r.get("id") in retained_ids:
                continue
            combined.append(r)
        return combined

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


    def run_research(self, topic: str, silent_plan: bool = False):
        self.current_topic = topic
        self.results = []
        self.synthesis = ""

        # 1. Planning phase
        with self.console.status(f"[bold yellow]Planning research for: {topic}...[/]", spinner="dots"):
            self.plan = self.planner.generate_plan(topic, mode=self.mode, context=self.retained_results)

        if silent_plan:
            self.execute_search()
        else:
            self.interactive_planning()

    def run_tree_research(self, topic: str, max_depth: int = 1, current_depth: int = 0):
        """
        Recursive research tree implementation.
        """
        indent = "  " * current_depth
        self.console.print(f"\n{indent}[bold cyan]🌲 Researching Branch (Depth {current_depth}): {topic}[/]")
        
        # 1. Standard research run
        # We skip interactive planning for sub-branches to keep it moving, 
        # but the first root topic can be interactive if we call it from start()
        self.run_research(topic, silent_plan=(current_depth > 0))
        
        if not self.synthesis:
            self.console.print(f"{indent}[red]Failed to synthesize results for {topic}.[/]")
            return

        # 2. Save this specific topic's report
        self.save_tree_report(topic, current_depth)

        # 3. Recursive expansion
        if current_depth < max_depth:
            with self.console.status(f"{indent}[bold magenta]Identifying expansion topics...[/]"):
                sub_topics = self.planner.identify_expansion_topics(topic, self.synthesis)
            
            if not sub_topics:
                self.console.print(f"{indent}[yellow]No further expansion topics identified for {topic}.[/]")
                return

            # Interactive selection
            try:
                import questionary
                selected = questionary.checkbox(
                    f"Select sub-topics to expand from '{topic}':",
                    choices=[questionary.Choice(t, checked=False) for t in sub_topics]
                ).ask()
            except ImportError:
                self.console.print(f"{indent}[yellow]Questionary not found. Expanding into all topics: {', '.join(sub_topics)}[/]")
                selected = sub_topics

            if selected:
                for sub in selected:
                    # Clear results for next run but keep context? 
                    # For now, let's keep it isolated but maybe pass some context?
                    # The user wants "research_topic.md", so isolation is better.
                    self.run_tree_research(sub, max_depth, current_depth + 1)

    def save_tree_report(self, topic: str, depth: int):
        safe_topic = "".join([c if c.isalnum() else "_" for c in topic])
        export_dir = os.path.join(DATA_DIR, "research_trees", self.safe_root_topic if hasattr(self, "safe_root_topic") else safe_topic)
        os.makedirs(export_dir, exist_ok=True)
        
        file_path = os.path.join(export_dir, f"{'sub_' * depth}{safe_topic}.md")
        
        md_lines = [
            f"# Research Report: {topic}",
            f"**Depth Level:** {depth}",
            f"**Generated At:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Synthesis",
            self.synthesis,
            "\n## Sources Considered",
        ]
        
        for idx, r in enumerate(self.results[:10], 1):
            title = r.get("title", "(no title)")
            link = r.get("link", "")
            snippet = r.get("snippet", "")
            md_lines.append(f"### {idx}. {title}")
            if link: md_lines.append(f"**Link:** {link}")
            md_lines.append(f"\n{snippet}\n")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        
        self.console.print(f"[bold green]Report saved:[/] [cyan]{file_path}[/]")

    def interactive_planning(self):
        while True:
            self.console.print(Panel(
                f"[bold green]Objective:[/] {self.plan['objective']}\n"
                f"[bold cyan]Queries:[/] {', '.join(self._get_query_labels(self.plan['queries']))}",
                title="Proposed Research Plan", border_style="green"
            ))

            choice = self._ask(
                "\n[bold]Execute plan, [yellow]refine[/] it, or [red]cancel[/]?[/]",
                choices=["e", "r", "c"], default="e"
            )

            if choice == "e":
                self.execute_search()
                break
            elif choice == "r":
                feedback = self._ask("\n[yellow]What should I change in the plan?[/]")
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

    def _ensure_engine(self):
        if not HAS_HYBRID_ENGINE:
            self.console.print("[yellow]Warning: HybridEngine not available (dependencies missing).[/]")
            return False
        if self.engine is None:
            with self.console.status("[bold magenta]Initializing Local RAG Engine...[/]"):
                self.engine = HybridEngine()
        return True

    def execute_search(self):
        # 2. Searching phase
        with self.console.status(f"[bold yellow]Searching {self.mode.upper()}...[/]", spinner="earth"):
            if self.mode == "arxiv":
                self.results = self.arxiv_searcher.search(self.plan["queries"])
            elif self.mode == "web":
                self.results = self.web_searcher.search_text(self.plan["queries"])
            elif self.mode == "news":
                self.results = self.web_searcher.search_news(self.plan["queries"])
            elif self.mode == "wiki":
                self.results = self.web_searcher.search_wikipedia(self.plan["queries"])
            elif self.mode == "local":
                if self._ensure_engine():
                    import asyncio
                    local_res = []
                    for q in self.plan["queries"]:
                        # HybridEngine.get_context_async returns a dict
                        ctx = asyncio.run(self.engine.get_context_async(q))
                        # Transform RAG context to research result format
                        for text, source in ctx.get("vector_context", []):
                            local_res.append({
                                "title": f"Local Context: {source}",
                                "snippet": text,
                                "source": "local",
                                "link": source
                            })
                    self.results = local_res
            elif self.mode == "deep":
                arxiv_queries = [q["query"] for q in self.plan["queries"] if isinstance(q, dict) and q.get("backend") == "arxiv"]
                web_queries = [q["query"] for q in self.plan["queries"] if isinstance(q, dict) and q.get("backend") == "web"]
                news_queries = [q["query"] for q in self.plan["queries"] if isinstance(q, dict) and q.get("backend") == "news"]
                wiki_queries = [q["query"] for q in self.plan["queries"] if isinstance(q, dict) and q.get("backend") == "wiki"]
                local_queries = [q["query"] for q in self.plan["queries"] if isinstance(q, dict) and q.get("backend") == "local"]
                
                arxiv_res = self.arxiv_searcher.search(arxiv_queries) if arxiv_queries else []
                web_res = self.web_searcher.search_text(web_queries) if web_queries else []
                news_res = self.web_searcher.search_news(news_queries) if news_queries else []
                wiki_res = self.web_searcher.search_wikipedia(wiki_queries) if wiki_queries else []
                
                local_res = []
                if local_queries and self._ensure_engine():
                    import asyncio
                    for q in local_queries:
                        ctx = asyncio.run(self.engine.get_context_async(q))
                        for text, source in ctx.get("vector_context", []):
                            local_res.append({
                                "title": f"Local: {source}",
                                "snippet": text,
                                "source": "local",
                                "link": source
                            })
                
                self.results = self._deduplicate(arxiv_res + web_res + news_res + wiki_res + local_res)

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
        if self.synthesis:
            self.console.print(Panel(
                self.synthesis, title=f"Research Report: {self.current_topic}", border_style="blue"
            ))

        # Display Results Table
        table = Table(title="Retrieved Sources (Memory + Current)", show_lines=True)
        table.add_column("Idx", justify="center", style="dim")
        table.add_column("Source", style="bold cyan")
        table.add_column("Title / Snippet", style="white")

        combined = self.get_combined_results()
        retained_links = {r.get("link") for r in self.retained_results if r.get("link")}
        retained_ids = {r.get("id") for r in self.retained_results if r.get("id")}

        for i, r in enumerate(combined, 1):
            source = r.get("source", "web").capitalize()
            is_retained = (r.get("link") in retained_links if r.get("link") else False) or \
                          (r.get("id") in retained_ids if r.get("id") else False)
            source_label = f"{source}\n[bold green][Retained][/]" if is_retained else source
            title = r.get("title", "(no title)")
            content = r.get("snippet", r.get("summary", ""))[:150] + "..."
            table.add_row(str(i), source_label, f"[bold]{title}[/]\n[dim]{content}[/]")

        self.console.print(table)
        self.console.print(
            "\n[dim]Use [bold]/ask[/] to query, [bold]/analyze <idx/selected>[/] for deep-dives, [bold]/select[/] to retain memory, [bold]/save[/] for KG, or [bold]/exit[/].[/]"
        )

    def chat_with_results(self, question: str):
        combined = self.get_combined_results()
        if not combined:
            self.console.print("[red]No research context available. Start a topic first.[/]")
            return
        
        with self.console.status("[bold cyan]Consulting findings...[/]", spinner="bouncingBall"):
            answer = self.planner.chat_with_results(self.current_topic, question, combined)
        
        self.console.print(Panel(answer, title="Research Assistant", border_style="cyan"))

    def analyze_item(self, idx_str: str):
        combined = self.get_combined_results()
        
        if idx_str.lower() in ["selected", "all"]:
            if not self.retained_results:
                self.console.print("[red]No retained sources to analyze.[/]")
                return
            for r in self.retained_results:
                with self.console.status(f"[bold magenta]Analyzing {r.get('title')}...[/]", spinner="pong"):
                    analysis = self.planner.analyze_result(self.current_topic, r)
                self.console.print(Panel(analysis, title=f"Deep Analysis: {r.get('title')}", border_style="magenta"))
            return

        try:
            indices = [int(x.strip()) - 1 for x in idx_str.split(",") if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(combined):
                    result = combined[idx]
                    with self.console.status(f"[bold magenta]Analyzing {result['title']}...[/]", spinner="pong"):
                        analysis = self.planner.analyze_result(self.current_topic, result)
                    self.console.print(Panel(analysis, title=f"Deep Analysis: {result['title']}", border_style="magenta"))
                else:
                    self.console.print(f"[red]Invalid index: {idx+1}[/]")
        except ValueError:
            self.console.print("[red]Please provide numeric indices (e.g. '1,3') or 'selected'.[/]")

    def interactive_select(self):
        try:
            import questionary
        except ImportError:
            self.console.print("[red]The 'questionary' library is not installed. Run 'pip install questionary'.[/]")
            return

        combined = self.get_combined_results()
        if not combined:
            self.console.print("[yellow]No results available to select.[/]")
            return

        retained_links = {r.get("link") for r in self.retained_results if r.get("link")}
        retained_ids = {r.get("id") for r in self.retained_results if r.get("id")}
        
        choices = []
        for i, r in enumerate(combined, 1):
            is_retained = (r.get("link") in retained_links if r.get("link") else False) or \
                          (r.get("id") in retained_ids if r.get("id") else False)
            source_label = r.get("source", "unknown").capitalize()
            title = r.get("title", "(no title)")[:80] + "..." if len(r.get("title", "")) > 80 else r.get("title", "(no title)")
            
            label = f"{i}. {title} [{source_label}]"
            choices.append(
                questionary.Choice(
                    title=label,
                    value=r,
                    checked=is_retained
                )
            )

        self.console.print("\n[bold cyan]Interactive Memory Selection[/]")
        self.console.print("[dim](Use ↑/↓ ARROW keys to move, SPACE to toggle, ENTER to confirm)[/]")
        
        selected_results = questionary.checkbox(
            "",
            choices=choices,
        ).ask()

        if selected_results is not None:
            self.retained_results = selected_results
            self.console.print(f"[bold green]Updated memory to {len(self.retained_results)} retained sources.[/]")

    def save_results(self, tags: List[str]):
        combined = self.get_combined_results()
        if not combined:
            self.console.print("[red]No results to save.[/]")
            return

        safe_topic = "".join([c if c.isalnum() else "_" for c in (self.current_topic or "untitled")])
        timestamp = int(time.time())
        export_dir = os.path.join(DATA_DIR, "saved_research", f"{safe_topic}_{timestamp}")
        
        os.makedirs(export_dir, exist_ok=True)
        file_path = os.path.join(export_dir, "research_report.md")
        
        md_lines = [
            f"# Research Topic: {self.current_topic or 'Untitled'}",
            f"**Tags:** {', '.join(tags) if tags else 'None'}\n",
        ]
        
        if self.synthesis:
            md_lines.extend(["## Executive Synthesis", self.synthesis, "\n"])
            
        md_lines.append("## Gathered Sources\n")
        
        grouped = {}
        for r in combined:
            src = r.get("source", "unknown").capitalize()
            if src not in grouped:
                grouped[src] = []
            grouped[src].append(r)
            
        for src, items in grouped.items():
            md_lines.append(f"### [{src}]")
            for idx, item in enumerate(items, 1):
                title = item.get("title", "(no title)")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                
                md_lines.append(f"#### {idx}. {title}")
                if link:
                    md_lines.append(f"**Link:** {link}")
                md_lines.append(f"\n{snippet}\n")
                
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
                
        self.console.print(f"[bold green]Saved {len(combined)} results from {len(grouped)} sources to:[/]\n[cyan]{file_path}[/]")

    # ------------------------------------------------------------------
    # Interactive CLI loop
    # ------------------------------------------------------------------

    def start(self):
        self.display_welcome()
        while True:
            try:
                self.console.print()
                cyan = "\001\033[1;36m\002"
                dim = "\001\033[2m\002"
                white = "\001\033[1;37m\002"
                reset = "\001\033[0m\002"
                prompt_str = f"{cyan}research{reset} {dim}({self.mode}){reset} {white}❯{reset} "
                
                try:
                    user_input = input(prompt_str).strip()
                except EOFError:
                    break

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
                            if new_mode in ["arxiv", "web", "news", "wiki", "local", "deep"]:
                                self.mode = new_mode
                                self.console.print(f"[green]Switched to [bold]{new_mode.upper()}[/] mode.[/]")
                            else:
                                self.console.print("[red]Invalid mode.[/]")
                    elif cmd == "/topic":
                        if len(parts) > 1:
                            self.run_research(parts[1])
                        else:
                            self.console.print("[red]Usage: /topic <topic>[/]")
                    elif cmd == "/tree":
                        if len(parts) > 1:
                            raw_args = parts[1].strip()
                            args = raw_args.rsplit(maxsplit=1)
                            if len(args) == 2 and args[1].isdigit():
                                topic = args[0].strip()
                                depth = int(args[1])
                            else:
                                topic = raw_args
                                depth = 1
                            
                            if not topic:
                                self.console.print("[red]Usage: /tree <topic> [depth][/]")
                                continue

                            self.safe_root_topic = "".join([c if c.isalnum() else "_" for c in topic])
                            self.run_tree_research(topic, depth)
                        else:
                            self.console.print("[red]Usage: /tree <topic> [depth][/]")
                    elif cmd == "/ask":
                        if len(parts) > 1:
                            self.chat_with_results(parts[1])
                        else:
                            self.console.print("[red]Usage: /ask <question>[/]")
                    elif cmd == "/analyze":
                        if len(parts) > 1:
                            self.analyze_item(parts[1])
                        else:
                            self.console.print("[red]Usage: /analyze <indices> or /analyze selected[/]")
                    elif cmd == "/select":
                        self.interactive_select()
                    elif cmd == "/refine":
                        if self.current_topic and self.plan:
                            feedback = parts[1] if len(parts) > 1 else self._ask("\n[yellow]Refinement feedback?[/]")
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
                    elif cmd == "/save":
                        if len(parts) > 1:
                            tags = [t.strip() for t in parts[1].split(",") if t.strip()]
                        else:
                            tags_input = self._ask("\n[yellow]Enter tags for these sources (comma-separated, or press Enter to skip)[/]")
                            tags = [t.strip() for t in tags_input.split(",") if t.strip()]
                        self.save_results(tags)
                    elif cmd == "/clear":
                        self.results = []
                        self.retained_results = []
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
    parser.add_argument("--mode", type=str, default="deep", help="Initial mode (deep, arxiv, web, wiki, news, local)")
    args = parser.parse_args()

    cli = ResearchCLI()
    cli.mode = args.mode
    if args.topic:
        cli.run_research(args.topic)
    else:
        cli.start()


if __name__ == "__main__":
    main()
