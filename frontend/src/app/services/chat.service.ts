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

    this.http.post<ChatResponse>(`${this.apiUrl}/chat`, request).subscribe({
      next: (res) => {
        const assistantMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: res.response,
          sources: res.sources,
          timestamp: new Date(),
          stats: res.stats,
          graphContext: res.graph_context,
        };
        this._addMessage(convId!, assistantMessage);

        // Update context stats on the conversation
        const stats = res.stats;
        if (stats) {
          this.conversations.update(cs =>
            cs.map(c =>
              c.id === convId
                ? {
                    ...c,
                    contextUtilization: stats.context_utilization ?? 0,
                    totalTokens: stats.total_tokens ?? 0,
                    contextWindow: stats.context_window ?? 8192,
                    updatedAt: new Date(),
                  }
                : c,
            ),
          );
        }

        this.isLoading.set(false);
        this._persist();

        // Generate AI title from the first user message if not yet generated
        if (isFirstMessage) {
          this._generateTitle(convId!, content.trim());
        }
      },
      error: (err) => {
        const errorMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content:
            'Sorry, something went wrong. Please make sure the API server is running and try again.',
          sources: [],
          timestamp: new Date(),
        };
        this._addMessage(convId!, errorMessage);
        this.isLoading.set(false);
        this._persist();
        console.error('Chat API error:', err);
      },
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
