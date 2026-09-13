import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '../core/auth/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  template: `
    <div class="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      <div class="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl">
        <div class="text-center mb-8">
          <div class="w-12 h-12 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-bold text-white text-xl mx-auto shadow-lg shadow-blue-500/30 mb-4">
            AI
          </div>
          <h1 class="text-xl font-bold text-white tracking-tight">Create Scientist Account</h1>
          <p class="text-xs text-slate-400 mt-1">Get started with autonomous ML experimentation</p>
        </div>

        <div *ngIf="errorMessage" class="mb-4 p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg">
          {{ errorMessage }}
        </div>

        <form (ngSubmit)="onSubmit()" class="space-y-4">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1.5">Username</label>
            <input
              type="text"
              [(ngModel)]="username"
              name="username"
              required
              class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500 transition"
              placeholder="data_scientist"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1.5">Email Address</label>
            <input
              type="email"
              [(ngModel)]="email"
              name="email"
              required
              class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500 transition"
              placeholder="scientist@deepmind.com"
            />
          </div>

          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1.5">Password</label>
            <input
              type="password"
              [(ngModel)]="password"
              name="password"
              required
              class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500 transition"
              placeholder="Min 8 chars, 1 uppercase, 1 digit"
            />
          </div>

          <button
            type="submit"
            [disabled]="loading"
            class="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-sm py-2.5 rounded-lg transition shadow-lg shadow-blue-500/20"
          >
            {{ loading ? 'Creating Account...' : 'Register' }}
          </button>
        </form>

        <p class="text-center text-xs text-slate-500 mt-6">
          Already have an account?
          <a routerLink="/auth/login" class="text-blue-400 hover:text-blue-300 font-medium">Sign in</a>
        </p>
      </div>
    </div>
  `
})
export class RegisterComponent {
  username = '';
  email = '';
  password = '';
  loading = false;
  errorMessage = '';

  constructor(private authService: AuthService, private router: Router) {}

  onSubmit(): void {
    if (!this.username || !this.email || !this.password) return;
    this.loading = true;
    this.errorMessage = '';

    this.authService.register({ username: this.username, email: this.email, password: this.password }).subscribe({
      next: () => {
        // Automatically login after successful registration
        this.authService.login({ email: this.email, password: this.password }).subscribe({
          next: () => {
            this.loading = false;
            this.router.navigate(['/dashboard']);
          },
          error: () => {
            this.loading = false;
            this.router.navigate(['/auth/login']);
          }
        });
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage = err.error?.detail || 'Registration failed. Check your password requirements.';
      }
    });
  }
}
