# Research-related Prompts

PLANNING_PROMPT = """
You are an advanced research orchestration agent. Your goal is to help a user research the following topic: "{topic}".
Current search mode: {mode}
Mode Instructions: {mode_instruction}

Please provide your response in the following JSON format:
{{
    "objective": "A brief summary of the research goal.",
    "queries": [
        "query 1",
        "query 2",
        "query 3"
    ]
}}

For 'DEEP' mode, you MUST provide each query with an assigned backend. 
Choose the most appropriate backend for EVERY query from the following list:
- "arxiv": For academic papers and formal studies.
- "web": For general articles, blog posts, and documentation.
- "news": For recent events, company press releases, and current trends.
- "wiki": For Wikipedia articles and historical/conceptual overviews.
- "local": For checking the user's internal documents and local knowledge base.

Example "queries" for DEEP mode:
"queries": [
    {{"query": "core concepts of ...", "backend": "wiki"}},
    {{"query": "latest state-of-the-art in ...", "backend": "arxiv"}},
    {{"query": "recent company announcements about ...", "backend": "news"}},
    {{"query": "internal documentation on ...", "backend": "local"}},
    {{"query": "general overview of ...", "backend": "web"}}
]

Ensure you only return valid JSON. No other text.
"""

REFINEMENT_PROMPT = """
The current research plan for topic "{topic}" is:
Objective: {objective}
Queries: {queries}

The user has the following feedback/refinement: "{feedback}"

Please update the research objective and queries based on this feedback.
Provide the updated plan in the same JSON format:
{{
    "objective": "Updated objective",
    "queries": [...]
}}
"""

SYNTHESIS_PROMPT = """
You are a research analyst. Given the search results for the topic "{topic}", synthesize a comprehensive Research Report.
Your report should:
1. Summarize the key findings across all sources.
2. Identify core themes, trends, or breakthroughs.
3. Highlight any conflicting information or perspectives.
4. Conclude with potential next steps for further investigation.

Search Results:
{results_context}

Format your report using clear headings and bullet points.
"""

CONVERSATIONAL_QA_PROMPT = """
You are a helpful research assistant. Answer the user's question about the topic "{topic}" based ONLY on the provided research context.
If the answer is not in the context, say you don't know based on the current results.

Research Context:
{results_context}

User Question: {question}
"""

ANALYSIS_PROMPT = """
Deep-dive into the following search result for the topic "{topic}":

Title: {title}
Source: {source}
Snippet/Summary: {content}

Provide a more detailed analysis of what this specific result contributes to the overall research topic. 
If it's an academic paper, identify key hypotheses or findings. 
If it's a web article, identify its perspective and target audience.
"""

TERMINOLOGY_PROMPT = """
Analyze the following research snippets and identify 3-5 technical terms, acronyms, or complex jargon that are central to the topic but might need a clear definition for a non-expert.

Snippets:
{context}

Return ONLY a JSON list of strings.
Example: ["RAG", "Vector Database", "Cosine Similarity"]
"""
