import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../core/services/api.service';
import { Experiment, ChatMessage } from '../core/models/types';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-[calc(100vh-8rem)] flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
      <!-- Chat Header -->
      <div class="p-4 border-b border-slate-800 bg-slate-950/40 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-sm font-bold text-blue-400">
            AI
          </div>
          <div>
            <h2 class="text-xs font-bold text-white">AI Data Scientist Conversation</h2>
            <p class="text-[10px] text-slate-400">Grounded against computed metrics, preprocessing logs, and SHAP explainability</p>
          </div>
        </div>

        <div class="flex items-center gap-2" *ngIf="experiments.length > 0">
          <label class="text-[11px] text-slate-400">Experiment:</label>
          <select
            [(ngModel)]="selectedExperimentId"
            (change)="onExperimentChange()"
            class="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
          >
            <option *ngFor="let exp of experiments" [value]="exp.id">{{ exp.name }}</option>
          </select>
        </div>
      </div>

      <!-- Messages Area -->
      <div class="flex-1 overflow-y-auto p-6 space-y-4 font-sans text-xs" #scrollContainer>
        <div *ngIf="messages.length === 0" class="text-center py-16 space-y-4">
          <div class="text-3xl font-bold text-blue-400">AI</div>
          <p class="text-sm font-semibold text-white">Ask your AI Data Scientist</p>
          <p class="text-xs text-slate-400 max-w-sm mx-auto">
            Get instant grounded answers regarding your dataset, feature distributions, model choices, or issue interactive retraining commands.
          </p>
          <div class="flex flex-wrap justify-center gap-2 max-w-md mx-auto pt-2">
            <button (click)="sendPreset('Why did you select the champion model?')" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-full text-[11px] border border-slate-700 transition">
              Why did you select the best model?
            </button>
            <button (click)="sendPreset('Which features are most important according to SHAP?')" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-full text-[11px] border border-slate-700 transition">
              Which features are most important?
            </button>
            <button (click)="sendPreset('Did the dataset contain missing values or leakage?')" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-full text-[11px] border border-slate-700 transition">
              Did the data contain missing values or leakage?
            </button>
          </div>
        </div>

        <div *ngFor="let msg of messages" [ngClass]="msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'">
          <div [ngClass]="msg.role === 'user' ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm ml-12' : 'bg-slate-950 border border-slate-800 text-slate-200 rounded-2xl rounded-tl-sm mr-12'" class="p-4 shadow-sm space-y-1 max-w-xl">
            <div class="flex items-center gap-2 mb-1">
              <span class="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                {{ msg.role === 'user' ? 'You' : 'Autonomous Scientist' }}
              </span>
              <span class="text-[9px] text-slate-500">{{ msg.created_at | date:'HH:mm' }}</span>
            </div>
            <p class="whitespace-pre-wrap leading-relaxed">{{ msg.content }}</p>
          </div>
        </div>

        <div *ngIf="loading" class="flex justify-start">
          <div class="bg-slate-950 border border-slate-800 text-slate-400 p-4 rounded-2xl rounded-tl-sm flex items-center gap-2 text-xs">
            <span class="inline-block w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            AI Scientist is consulting experiment facts and reasoning...
          </div>
        </div>
      </div>

      <!-- Input Bar -->
      <div class="p-4 border-t border-slate-800 bg-slate-950/40">
        <form (ngSubmit)="sendMessage()" class="flex items-center gap-2">
          <input
            type="text"
            [(ngModel)]="userInput"
            name="chatInput"
            [disabled]="loading || !selectedExperimentId"
            class="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500 transition"
            placeholder="Ask a question or issue a command (e.g., 'Retrain with higher regularization')..."
          />
          <button
            type="submit"
            [disabled]="loading || !userInput.trim() || !selectedExperimentId"
            class="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-5 py-3 rounded-xl text-xs font-semibold transition flex items-center gap-2 shrink-0 shadow-lg shadow-blue-500/20"
          >
            <span>Send</span>
          </button>
        </form>
      </div>
    </div>
  `
})
export class ChatComponent implements OnInit {
  experiments: Experiment[] = [];
  selectedExperimentId = '';
  messages: ChatMessage[] = [];
  userInput = '';
  loading = false;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getExperiments().subscribe(exps => {
      this.experiments = exps;
      if (exps.length > 0) {
        this.selectedExperimentId = exps[0].id;
        this.loadHistory();
      }
    });
  }

  onExperimentChange(): void {
    this.loadHistory();
  }

  loadHistory(): void {
    if (!this.selectedExperimentId) return;
    this.api.getChatHistory(this.selectedExperimentId).subscribe(msgs => {
      this.messages = msgs;
    });
  }

  sendPreset(prompt: string): void {
    this.userInput = prompt;
    this.sendMessage();
  }

  sendMessage(): void {
    if (!this.userInput.trim() || !this.selectedExperimentId) return;
    const msg = this.userInput.trim();
    this.userInput = '';
    this.loading = true;

    // Optimistically add user message
    this.messages.push({
      id: 'temp',
      role: 'user',
      content: msg,
      created_at: new Date().toISOString()
    });

    
    this.api.sendChatMessage(this.selectedExperimentId, msg).subscribe({
      next: (aiReply) => {
        this.loading = false;
        this.messages.push(aiReply);
      },
      error: () => {
        this.loading = false;
      }
    });
  }
}
