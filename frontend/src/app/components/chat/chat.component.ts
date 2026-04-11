import {
  Component,
  signal,
  inject,
  viewChild,
  ElementRef,
  ChangeDetectionStrategy,
  effect,
  OnInit,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { MessageBubbleComponent } from '../message-bubble/message-bubble.component';
import { ConversationSidebarComponent } from '../conversation-sidebar/conversation-sidebar.component';
import { LlmMode } from '../../models/chat.models';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, MessageBubbleComponent, ConversationSidebarComponent],
})
export class ChatComponent implements OnInit {
  readonly chatService = inject(ChatService);

  readonly messages = this.chatService.messages;
  readonly isLoading = this.chatService.isLoading;
  readonly llmMode = this.chatService.llmMode;
  readonly activeConversation = this.chatService.activeConversation;
  readonly quickPrompts = this.chatService.quickPrompts;
  readonly userInput = signal('');
  readonly isSettingsOpen = signal(false);

  readonly messagesContainer = viewChild<ElementRef<HTMLDivElement>>('messagesContainer');

  constructor() {
    // Auto-scroll when messages change
    effect(() => {
      this.messages(); // track signal
      this.scrollToBottom();
    });

    // Create a default conversation on first load if none exist
    effect(() => {
      const convs = this.chatService.conversations();
      if (convs.length === 0) {
        this.chatService.createConversation();
      }
    });
  }

  ngOnInit(): void {
    this.chatService.fetchQuickPrompts();
  }

  applyPrompt(text: string): void {
    if (this.isLoading()) return;
    this.userInput.set(text);
    this.sendMessage();
  }

  sendMessage(): void {
    const msg = this.userInput();
    if (!msg.trim()) return;
    this.chatService.sendMessage(msg);
    this.userInput.set('');
    setTimeout(() => this.scrollToBottom(), 100);
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  setMode(mode: LlmMode): void {
    this.chatService.llmMode.set(mode);
  }

  toggleSettings(): void {
    this.isSettingsOpen.update(v => !v);
  }

  updateSystemPrompt(event: Event): void {
    const val = (event.target as HTMLTextAreaElement).value;
    this.chatService.updateSystemPrompt(val);
  }

  clearChat(): void {
    if (confirm('Are you sure you want to clear all messages in this conversation?')) {
      this.chatService.clearMessages();
    }
  }

  handleRetry(msgId: string): void {
    this.chatService.retryMessage(msgId);
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const el = this.messagesContainer()?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    });
  }
}
