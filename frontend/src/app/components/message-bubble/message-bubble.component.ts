import { Component, input, output, ChangeDetectionStrategy } from '@angular/core';
import { ChatMessage } from '../../models/chat.models';
import { SourceBadgeComponent } from '../source-badge/source-badge.component';
import { DatePipe, DecimalPipe } from '@angular/common';

@Component({
  selector: 'app-message-bubble',
  templateUrl: './message-bubble.component.html',
  styleUrl: './message-bubble.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [SourceBadgeComponent, DatePipe, DecimalPipe],
})
export class MessageBubbleComponent {
  readonly message = input.required<ChatMessage>();
  readonly promptClick = output<string>();
}
