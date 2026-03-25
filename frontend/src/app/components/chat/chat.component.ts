import {
  Component,
  signal,
  inject,
  viewChild,
  ElementRef,
  ChangeDetectionStrategy,
  effect,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../../services/chat.service';
import { MessageBubbleComponent } from '../message-bubble/message-bubble.component';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrl: './chat.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, MessageBubbleComponent],
})
export class ChatComponent {
  private readonly chatService = inject(ChatService);

  readonly messages = this.chatService.messages;
  readonly isLoading = this.chatService.isLoading;
  readonly userInput = signal('');

  readonly messagesContainer = viewChild<ElementRef<HTMLDivElement>>('messagesContainer');

  constructor() {
    // Auto-scroll when messages change
    effect(() => {
      this.messages(); // track signal
      this.scrollToBottom();
    });
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

  private scrollToBottom(): void {
    // Use setTimeout to let the DOM update before scrolling
    setTimeout(() => {
      const el = this.messagesContainer()?.nativeElement;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    });
  }
}
