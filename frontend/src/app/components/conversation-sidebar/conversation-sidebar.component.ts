import {
  Component,
  inject,
  signal,
  ChangeDetectionStrategy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatePipe } from '@angular/common';
import { ChatService } from '../../services/chat.service';
import { Conversation } from '../../models/chat.models';

@Component({
  selector: 'app-conversation-sidebar',
  templateUrl: './conversation-sidebar.component.html',
  styleUrl: './conversation-sidebar.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, DatePipe],
})
export class ConversationSidebarComponent {
  readonly chatService = inject(ChatService);

  readonly conversations = this.chatService.conversations;
  readonly activeConversationId = this.chatService.activeConversationId;

  /** ID of the conversation currently being renamed */
  readonly editingId = signal<string | null>(null);
  readonly editingName = signal('');

  newConversation(): void {
    this.chatService.createConversation();
  }

  selectConversation(id: string): void {
    this.chatService.switchConversation(id);
  }

  deleteConversation(event: Event, id: string): void {
    event.stopPropagation();
    this.chatService.deleteConversation(id);
  }

  startRename(event: Event, conv: Conversation): void {
    event.stopPropagation();
    this.editingId.set(conv.id);
    this.editingName.set(conv.name);
  }

  commitRename(id: string): void {
    const name = this.editingName().trim();
    if (name) {
      this.chatService.renameConversation(id, name);
    }
    this.editingId.set(null);
  }

  cancelRename(): void {
    this.editingId.set(null);
  }

  onRenameKeyDown(event: KeyboardEvent, id: string): void {
    if (event.key === 'Enter') this.commitRename(id);
    if (event.key === 'Escape') this.cancelRename();
  }


  trackById(_: number, c: Conversation): string {
    return c.id;
  }
}
