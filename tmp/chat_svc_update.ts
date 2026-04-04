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
      });
      this.isLoading.set(false);
      this._persist();
      console.error('Chat API error:', err);
    });
