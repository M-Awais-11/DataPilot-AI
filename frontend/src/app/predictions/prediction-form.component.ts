import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../core/services/api.service';

@Component({
  selector: 'app-prediction-form',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 class="text-xl font-bold text-white tracking-tight">Prediction API Testing</h1>
        <p class="text-xs text-slate-400 mt-1">Serve real-time inference with local SHAP feature contribution explanations</p>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
        <div>
          <div class="flex items-center justify-between mb-2">
            <label class="block text-xs font-medium text-slate-400">Input JSON Features</label>
            <button (click)="loadSample()" class="text-xs text-blue-400 hover:text-blue-300">
              Load Sample Data
            </button>
          </div>
          <textarea
            [(ngModel)]="featuresJson"
            rows="8"
            class="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 font-mono text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500 transition"
            placeholder='{\n  "Age": 35,\n  "Fare": 53.1,\n  "Pclass": 1\n}'
          ></textarea>
        </div>

        <button
          (click)="predict()"
          [disabled]="loading"
          class="w-full bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-medium text-xs py-2.5 rounded-lg transition shadow-lg shadow-purple-500/20 flex items-center justify-center gap-2"
        >
          <span>{{ loading ? 'Running Inference...' : 'Predict with Production Model' }}</span>
        </button>

        <!-- Prediction Result Card -->
        <div *ngIf="result" class="mt-6 p-5 bg-slate-950 border border-purple-500/30 rounded-xl space-y-4">
          <div class="flex items-center justify-between">
            <div>
              <span class="text-[10px] text-purple-400 uppercase font-bold tracking-wider">Inference Output</span>
              <p class="text-2xl font-extrabold text-white mt-1">{{ result.prediction }}</p>
            </div>
            <div class="text-right" *ngIf="result.confidence">
              <span class="text-[10px] text-slate-400 uppercase">Confidence</span>
              <p class="text-lg font-bold text-emerald-400 mt-1">{{ (result.confidence * 100).toFixed(1) }}%</p>
            </div>
          </div>

          <div class="text-xs text-slate-400 flex items-center gap-2 border-t border-slate-900 pt-3">
            <span>Model: <b class="text-white">{{ result.algorithm }}</b></span>
            <span>/</span>
            <span>Version: <b class="text-white">{{ result.model_version }}</b></span>
          </div>

          <!-- Local SHAP Contributions -->
          <div *ngIf="result.shap_contributions" class="border-t border-slate-900 pt-3 space-y-2">
            <p class="text-xs font-semibold text-slate-300">Feature Contribution Breakdown (Local SHAP):</p>
            <div *ngFor="let item of getShapItems()" class="flex items-center justify-between text-xs font-mono">
              <span class="text-slate-400">{{ item.key }}</span>
              <span [ngClass]="item.val >= 0 ? 'text-emerald-400' : 'text-rose-400'">
                {{ item.val >= 0 ? '+' : '' }}{{ item.val }}
              </span>
            </div>
          </div>
        </div>

        <div *ngIf="errorMessage" class="p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg">
          {{ errorMessage }}
        </div>
      </div>
    </div>
  `
})
export class PredictionFormComponent {
  featuresJson = '{\n  "Age": 28,\n  "Fare": 75.25,\n  "Pclass": 1\n}';
  loading = false;
  result: any = null;
  errorMessage = '';

  constructor(private api: ApiService) {}

  loadSample(): void {
    this.featuresJson = JSON.stringify({
      Age: 32,
      Fare: 45.0,
      Pclass: 2,
      SibSp: 0,
      Parch: 0
    }, null, 2);
  }

  predict(): void {
    this.errorMessage = '';
    this.result = null;
    let parsed: any;
    try {
      parsed = JSON.parse(this.featuresJson);
    } catch {
      this.errorMessage = 'Invalid JSON syntax. Please verify input.';
      return;
    }

    this.loading = true;
    this.api.predictProduction(parsed).subscribe({
      next: (res) => {
        this.loading = false;
        this.result = res;
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage = err.error?.detail || 'Inference failed. Check if a model has been promoted to production.';
      }
    });
  }

  getShapItems(): { key: string; val: number }[] {
    if (!this.result?.shap_contributions) return [];
    return Object.entries(this.result.shap_contributions).map(([key, val]) => ({
      key,
      val: val as number
    }));
  }
}
