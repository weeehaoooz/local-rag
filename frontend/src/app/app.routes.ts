import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./components/chat/chat.component').then(m => m.ChatComponent),
  },
  {
    path: '**',
    redirectTo: '',
  },
];
