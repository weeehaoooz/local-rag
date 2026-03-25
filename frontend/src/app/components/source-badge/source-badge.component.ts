import { Component, input, ChangeDetectionStrategy } from '@angular/core';
import { ChatSource } from '../../models/chat.models';

@Component({
  selector: 'app-source-badge',
  templateUrl: './source-badge.component.html',
  styleUrl: './source-badge.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SourceBadgeComponent {
  readonly source = input.required<ChatSource>();
}
