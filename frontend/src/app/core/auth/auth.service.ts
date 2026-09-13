import { Injectable, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';
import { User, AuthResponse } from '../models/types';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = 'http://localhost:8000/api/auth';
  
  // Angular Signals for reactive state
  currentUser = signal<User | null>(this.getStoredUser());
  token = signal<string | null>(localStorage.getItem('token'));
  isAuthenticated = computed(() => !!this.token());

  constructor(private http: HttpClient, private router: Router) {
    if (this.token()) {
      this.loadUserProfile();
    }
  }

  private getStoredUser(): User | null {
    const raw = localStorage.getItem('user');
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  register(data: any): Observable<User> {
    return this.http.post<User>(`${this.apiUrl}/register`, data);
  }

  login(data: any): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/login`, data).pipe(
      tap(res => {
        localStorage.setItem('token', res.access_token);
        localStorage.setItem('refresh_token', res.refresh_token);
        this.token.set(res.access_token);
        this.loadUserProfile();
      })
    );
  }

  loadUserProfile(): void {
    this.http.get<User>(`${this.apiUrl}/me`).subscribe({
      next: (user) => {
        this.currentUser.set(user);
        localStorage.setItem('user', JSON.stringify(user));
      },
      error: (err) => {
        // Only logout on genuine auth failures (invalid/expired token).
        // Connection errors (status 0) or server errors (5xx) should NOT
        // clear a valid token — the user is still logged in.
        if (err.status === 401 || err.status === 403) {
          const refreshToken = localStorage.getItem('refresh_token');
          if (refreshToken) {
            this.refreshToken(refreshToken).subscribe({
              next: (res) => {
                localStorage.setItem('token', res.access_token);
                if (res.refresh_token) {
                  localStorage.setItem('refresh_token', res.refresh_token);
                }
                this.token.set(res.access_token);
                this.loadUserProfile();
              },
              error: () => this.logout()
            });
          } else {
            this.logout();
          }
        }
      }
    });
  }

  refreshToken(refreshToken: string): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/refresh`, { refresh_token: refreshToken });
  }

  logout(): void {
    localStorage.removeItem('token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    this.token.set(null);
    this.currentUser.set(null);
    this.router.navigate(['/auth/login']);
  }
}
