import { Injectable, signal, computed, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import {
  ChatMessage,
  ChatRequest,
  ChatResponse,
  Conversation,
  LlmMode,
  TitleRequest,
  TitleResponse,
  SuggestionResponse,
} from '../models/chat.models';

const STORAGE_KEY = 'kg_rag_conversations';
const ACTIVE_ID_KEY = 'kg_rag_active_conversation';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiUrl = 'http://localhost:8000/api';
  private readonly http = inject(HttpClient);

  /** All conversations */
  readonly conversations = signal<Conversation[]>(this._loadFromStorage());

  /** Active conversation ID */
  readonly activeConversationId = signal<string | null>(
    this._readActiveId(),
  );

  /** LLM mode for next message */
  readonly llmMode = signal<LlmMode>('fast');

  /** Whether a request is in-flight */
  readonly isLoading = signal(false);

  /** Quick prompts fetched from KG */
  readonly quickPrompts = signal<string[]>([]);

  /** Active conversation object */
  readonly activeConversation = computed(() => {
    const id = this.activeConversationId();
    return this.conversations().find(c => c.id === id) ?? null;
  });

  /** Messages of the active conversation */
  readonly messages = computed(() => this.activeConversation()?.messages ?? []);

  // ── Conversation management ──────────────────────────────────────────

  createConversation(): string {
    const count = this.conversations().length + 1;
    const conv: Conversation = {
      id: crypto.randomUUID(),
      name: `Conversation ${count}`,
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
      contextUtilization: 0,
      totalTokens: 0,
      contextWindow: 8192,
      titleGenerated: false,
      systemPrompt: '',
    };
    this.conversations.update(cs => [conv, ...cs]);
    this.activeConversationId.set(conv.id);
    this._persist();
    return conv.id;
  }

  switchConversation(id: string): void {
    this.activeConversationId.set(id);
    localStorage.setItem(ACTIVE_ID_KEY, id);
  }

  deleteConversation(id: string): void {
    this.conversations.update(cs => cs.filter(c => c.id !== id));

    // If we deleted the active one, switch to the first available
    if (this.activeConversationId() === id) {
      const remaining = this.conversations();
      const nextId = remaining.length > 0 ? remaining[0].id : null;
      this.activeConversationId.set(nextId);
      if (nextId) localStorage.setItem(ACTIVE_ID_KEY, nextId);
      else localStorage.removeItem(ACTIVE_ID_KEY);
    }
    this._persist();
  }

  renameConversation(id: string, newName: string): void {
    this.conversations.update(cs =>
      cs.map(c => (c.id === id ? { ...c, name: newName, titleGenerated: true } : c)),
    );
    this._persist();
  }

  clearMessages(): void {
    const id = this.activeConversationId();
    if (!id) return;
    this.conversations.update(cs =>
      cs.map(c =>
        c.id === id
          ? { ...c, messages: [], contextUtilization: 0, totalTokens: 0, titleGenerated: false }
          : c,
      ),
    );
    this._persist();
  }

  updateSystemPrompt(prompt: string): void {
    const id = this.activeConversationId();
    if (!id) return;
    this.conversations.update(cs =>
      cs.map(c =>
        c.id === id ? { ...c, systemPrompt: prompt, updatedAt: new Date() } : c
      )
    );
    this._persist();
  }

  // ── Messaging ────────────────────────────────────────────────────────

  sendMessage(content: string): void {
    if (!content.trim() || this.isLoading()) return;

    // Ensure there is an active conversation
    let convId = this.activeConversationId();
    if (!convId) {
      convId = this.createConversation();
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: content.trim(),
      sources: [],
      timestamp: new Date(),
    };

    this._addMessage(convId, userMessage);
    this.isLoading.set(true);

    const isFirstMessage = this.activeConversation()!.messages.length === 1;

    const history = this.activeConversation()!.messages.slice(0, -1);
    const systemPrompt = this.activeConversation()!.systemPrompt;
    
    const request: ChatRequest = { 
      message: content.trim(), 
      mode: this.llmMode(), 
      history,
      system_prompt: systemPrompt
    };

    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '', // Waiting for response
      sources: [],
      timestamp: new Date(),
      isStreaming: true,
      processingState: 'Initializing...',
      liveTokenCount: 0
    };
    this._addMessage(convId!, assistantMessage);

    fetch(`${this.apiUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    }).then(async response => {
      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      if (!reader) return;

      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (part.startsWith('data: ')) {
            const dataStr = part.substring(6);
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'status') {
                this._updateMessage(convId!, assistantMessage.id, {
                  processingState: data.message,
                  liveTokenCount: data.tokens || 0
                });
              } else if (data.type === 'done') {
                const res = data.response;
                this._updateMessage(convId!, assistantMessage.id, {
                  isStreaming: false,
                  processingState: undefined,
                  liveTokenCount: undefined,
                  content: res.response,
                  sources: res.sources,
                  stats: res.stats,
                  graphContext: res.graph_context,
                  suggestedPrompts: res.suggested_prompts,
                  sub_queries: res.sub_queries,
                  reflection_loops: res.reflection_loops,
                  retrieval_grade: res.retrieval_grade,
                  answer_grade: res.answer_grade,
                  tools_used: res.tools_used,
                  orchestrator_rationale: res.orchestrator_rationale,
                });
                
                if (res.stats) {
                  this.conversations.update(cs =>
                    cs.map(c =>
                      c.id === convId
                        ? {
                            ...c,
                            contextUtilization: res.stats.context_utilization ?? 0,
                            totalTokens: res.stats.total_tokens ?? 0,
                            contextWindow: res.stats.context_window ?? 8192,
                            updatedAt: new Date(),
                          }
                        : c,
                    ),
                  );
                }
                this._persist();
                if (isFirstMessage) this._generateTitle(convId!, content.trim());
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
      this.isLoading.set(false);
    }).catch(err => {
      this._updateMessage(convId!, assistantMessage.id, {
        content: 'Sorry, something went wrong. Please make sure the API server is running and try again.',
        isStreaming: false,
        processingState: undefined,
        error: true,
      });
      this.isLoading.set(false);
      this._persist();
      console.error('Chat API error:', err);
    });
  }

  retryMessage(assistantMsgId: string): void {
    const convId = this.activeConversationId();
    if (!convId) return;

    const conv = this.activeConversation();
    if (!conv) return;

    const assistantIdx = conv.messages.findIndex(m => m.id === assistantMsgId);
    if (assistantIdx === -1) return;

    const userMsg = conv.messages[assistantIdx - 1];
    if (!userMsg || userMsg.role !== 'user') return;

    // Remove the failed assistant message
    this.conversations.update(cs =>
      cs.map(c =>
        c.id === convId
          ? { ...c, messages: c.messages.filter(m => m.id !== assistantMsgId) }
          : c
      )
    );

    // Re-send the user message
    this.sendMessage(userMsg.content);
  }

  fetchQuickPrompts(): void {
    this.http.get<SuggestionResponse>(`${this.apiUrl}/suggestions`).subscribe({
      next: (res) => this.quickPrompts.set(res.suggestions || []),
      error: (err) => {
        console.error('Failed to load suggestions', err);
        this.quickPrompts.set([]);
      }
    });
  }

  // ── Private helpers ──────────────────────────────────────────────────

  private _addMessage(convId: string, msg: ChatMessage): void {
    this.conversations.update(cs =>
      cs.map(c =>
        c.id === convId
          ? { ...c, messages: [...c.messages, msg], updatedAt: new Date() }
          : c,
      ),
    );
  }

  private _updateMessage(convId: string, msgId: string, updates: Partial<ChatMessage>): void {
    this.conversations.update(cs =>
      cs.map(c =>
        c.id === convId
          ? {
              ...c,
              messages: c.messages.map(m => (m.id === msgId ? { ...m, ...updates } : m)),
              updatedAt: new Date(),
            }
          : c,
      ),
    );
  }

  private _generateTitle(convId: string, firstMessage: string): void {
    const req: TitleRequest = { first_message: firstMessage };
    this.http.post<TitleResponse>(`${this.apiUrl}/title`, req).subscribe({
      next: (res) => {
        if (res.title) {
          this.conversations.update(cs =>
            cs.map(c =>
              c.id === convId ? { ...c, name: res.title, titleGenerated: true } : c,
            ),
          );
          this._persist();
        }
      },
      error: () => {
        // Non-critical – keep the default name
      },
    });
  }

  private _persist(): void {
    try {
      const raw = this.conversations();
      localStorage.setItem(STORAGE_KEY, JSON.stringify(raw));
      const activeId = this.activeConversationId();
      if (activeId) localStorage.setItem(ACTIVE_ID_KEY, activeId);
    } catch {
      // Storage quota exceeded or unavailable – ignore
    }
  }

  private _loadFromStorage(): Conversation[] {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed: Conversation[] = JSON.parse(raw);
      // Re-hydrate Date objects
      return parsed.map(c => ({
        ...c,
        createdAt: new Date(c.createdAt),
        updatedAt: new Date(c.updatedAt),
        messages: c.messages.map(m => ({ ...m, timestamp: new Date(m.timestamp) })),
      }));
    } catch {
      return [];
    }
  }

  private _readActiveId(): string | null {
    try {
      return localStorage.getItem(ACTIVE_ID_KEY);
    } catch {
      return null;
    }
  }
}
