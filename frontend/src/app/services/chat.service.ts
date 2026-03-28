import { Injectable, signal, computed, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ChatMessage, ChatRequest, ChatResponse } from '../models/chat.models';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly apiUrl = 'http://localhost:8000/api';
  private readonly http = inject(HttpClient);

  /** All messages in the conversation */
  readonly messages = signal<ChatMessage[]>([]);

  /** Whether a request is in-flight */
  readonly isLoading = signal(false);

  /** Number of messages */
  readonly messageCount = computed(() => this.messages().length);

  sendMessage(content: string): void {
    if (!content.trim() || this.isLoading()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: content.trim(),
      sources: [],
      timestamp: new Date(),
    };

    this.messages.update(msgs => [...msgs, userMessage]);
    this.isLoading.set(true);

    const request: ChatRequest = { message: content.trim() };

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
        this.messages.update(msgs => [...msgs, assistantMessage]);
        this.isLoading.set(false);
      },
      error: (err) => {
        const errorMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: 'Sorry, something went wrong. Please make sure the API server is running and try again.',
          sources: [],
          timestamp: new Date(),
        };
        this.messages.update(msgs => [...msgs, errorMessage]);
        this.isLoading.set(false);
        console.error('Chat API error:', err);
      },
    });
  }

  clearMessages(): void {
    this.messages.set([]);
  }
}
