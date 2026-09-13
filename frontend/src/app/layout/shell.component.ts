import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '../core/auth/auth.service';

@Component({
  selector: 'app-shell',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <div class="app-shell">
      <aside class="app-sidebar">
        <a routerLink="/dashboard" class="brand">
          <span class="brand-mark" style="background:linear-gradient(135deg,#3b82f6,#1d4ed8); font-size:13px; font-weight:800; display:flex; align-items:center; justify-content:center; letter-spacing:-0.5px;">DP</span>
          <span><strong>DataPilot <em style="color:#60a5fa; font-style:normal;">AI</em></strong><small>Autonomous Data Science</small></span>
        </a>

        <nav class="main-nav" aria-label="Primary navigation">
          <a routerLink="/" routerLinkActive="is-active" [routerLinkActiveOptions]="{exact: true}"><span>00</span>Home Page</a>
          <a routerLink="/dashboard" routerLinkActive="is-active"><span>01</span>Overview</a>
          <a routerLink="/datasets" routerLinkActive="is-active"><span>02</span>Datasets</a>
          <a routerLink="/experiments" routerLinkActive="is-active"><span>03</span>Analyses</a>
          <a routerLink="/models" routerLinkActive="is-active"><span>04</span>Model registry</a>
          <a routerLink="/predictions" routerLinkActive="is-active"><span>05</span>Predictions</a>
          <a routerLink="/reports" routerLinkActive="is-active"><span>06</span>Reports</a>
          <a routerLink="/chat" routerLinkActive="is-active"><span>07</span>Assistant</a>
        </nav>

        <div class="sidebar-user">
          <div class="user-avatar">{{ authService.currentUser()?.username?.[0]?.toUpperCase() || 'U' }}</div>
          <div>
            <strong>{{ authService.currentUser()?.username || 'Scientist' }}</strong>
            <small>{{ authService.currentUser()?.email || 'Workspace account' }}</small>
          </div>
          <button (click)="logout()" title="Sign out">Sign out</button>
        </div>
      </aside>

      <div class="app-workspace">
        <header class="app-topbar">
          <div class="system-status"><i></i>System ready</div>
          <a routerLink="/experiments/new" class="button button-primary">Start new analysis</a>
        </header>
        <main class="app-main"><router-outlet></router-outlet></main>
      </div>
    </div>
  `,
})
export class ShellComponent {
  constructor(public authService: AuthService, private router: Router) {}

  logout(): void {
    this.authService.logout();
    this.router.navigate(['/auth/login']);
  }
}
