export interface ChatSource {
  title: string;
  category: string;
  file: string;
}

export interface GraphNode {
  text: string;
  source: string;
}

export interface ChatStats {
  tps: number;
  context_utilization: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: ChatSource[];
  timestamp: Date;
  stats?: ChatStats;
  graphContext?: GraphNode[];
}

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  response: string;
  sources: ChatSource[];
  stats: ChatStats;
  graph_context: GraphNode[];
}
