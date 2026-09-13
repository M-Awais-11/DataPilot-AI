import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { ApiService } from '../core/services/api.service';
import { Dataset, DatasetProfile } from '../core/models/types';

@Component({
  selector: 'app-new-experiment',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  template: `
    <div class="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 class="text-xl font-bold text-white tracking-tight">Launch Autonomous Experiment</h1>
        <p class="text-xs text-slate-400 mt-1">Configure and start the multi-agent ML workflow</p>
      </div>

      <div *ngIf="errorMessage" class="p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg">
        {{ errorMessage }}
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
        <!-- Dataset Selection -->
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1.5">Select Dataset</label>
          <select
            [(ngModel)]="selectedDatasetId"
            (change)="onDatasetChange()"
            class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-slate-100 focus:outline-none focus:border-blue-500 transition"
          >
            <option value="" disabled>-- Select a dataset --</option>
            <option *ngFor="let ds of datasets" [value]="ds.id">
              {{ ds.name }} ({{ ds.row_count ? (ds.row_count | number) + ' rows' : 'Profiling' }})
            </option>
          </select>
        </div>

        <!-- Experiment Name -->
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1.5">Experiment Name</label>
          <input
            type="text"
            [(ngModel)]="experimentName"
            class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500 transition"
            placeholder="e.g. Disease Prediction Run 1"
          />
        </div>

        <!-- Target Column (Optional Override) -->
        <div>
          <div class="flex items-center justify-between mb-1.5">
            <label class="block text-xs font-medium text-slate-400">Target Column</label>
            <span class="text-[10px] text-blue-400" *ngIf="suggestedTarget">
              Suggested: <b>{{ suggestedTarget }}</b>
            </span>
          </div>
          <input
            type="text"
            [(ngModel)]="targetColumn"
            class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500 transition"
            [placeholder]="suggestedTarget || 'Leave blank for autonomous detection'"
          />
        </div>

        <!-- Primary Metric -->
        <div>
          <div class="flex items-center justify-between mb-1.5">
            <label class="block text-xs font-medium text-slate-400">Primary Optimization Metric</label>
            <span class="text-[10px] text-purple-400 font-semibold uppercase" *ngIf="suggestedProblemType">
              Task: {{ suggestedProblemType }}
            </span>
          </div>
          <select
            [(ngModel)]="primaryMetric"
            class="w-full bg-slate-950 border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-slate-100 focus:outline-none focus:border-blue-500 transition"
          >
            <optgroup label="Regression Metrics" *ngIf="suggestedProblemType === 'regression'">
              <option value="rmse">Root Mean Squared Error (RMSE) - Standard Regression</option>
              <option value="r2">R2 Score - Explained Variance</option>
              <option value="mae">Mean Absolute Error (MAE) - Robust to Outliers</option>
            </optgroup>
            <optgroup label="Classification Metrics" *ngIf="suggestedProblemType !== 'regression'">
              <option value="f1_weighted">F1 Score (Weighted) - Recommended for Imbalanced Classification</option>
              <option value="accuracy">Accuracy - Standard Classification</option>
              <option value="roc_auc">ROC-AUC - Binary / Probabilistic</option>
            </optgroup>
            <optgroup label="All Metrics">
              <option value="rmse">RMSE (Regression)</option>
              <option value="r2">R2 (Regression)</option>
              <option value="mae">MAE (Regression)</option>
              <option value="f1_weighted">F1 Weighted (Classification)</option>
              <option value="accuracy">Accuracy (Classification)</option>
              <option value="roc_auc">ROC-AUC (Classification)</option>
            </optgroup>
          </select>
        </div>

        <button
          (click)="launch()"
          [disabled]="loading || !selectedDatasetId || !experimentName"
          class="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-xs py-3 rounded-lg transition shadow-lg shadow-blue-500/20 flex items-center justify-center gap-2"
        >
          <span>{{ loading ? 'Starting Autonomous Pipeline...' : 'Start Autonomous Data Scientist' }}</span>
        </button>
      </div>
    </div>
  `
})
export class NewExperimentComponent implements OnInit {
  datasets: Dataset[] = [];
  selectedDatasetId = '';
  experimentName = '';
  targetColumn = '';
  suggestedTarget = '';
  suggestedProblemType = '';
  primaryMetric = 'f1_weighted';
  loading = false;
  errorMessage = '';

  constructor(private api: ApiService, private route: ActivatedRoute, private router: Router) {}

  ngOnInit(): void {
    const datasetIdParam = this.route.snapshot.queryParamMap.get('datasetId');
    this.api.getDatasets().subscribe(ds => {
      this.datasets = ds;
      if (datasetIdParam) {
        this.selectedDatasetId = datasetIdParam;
        this.onDatasetChange();
      }
    });
  }

  onDatasetChange(): void {
    const ds = this.datasets.find(d => d.id === this.selectedDatasetId);
    if (ds) {
      this.experimentName = `${ds.name} Analysis`;
      this.api.getDatasetProfile(ds.id).subscribe(p => {
        const profile = p?.profile_json;
        if (profile?.target_column_suggestion) {
          this.suggestedTarget = profile.target_column_suggestion;
          this.targetColumn = profile.target_column_suggestion;
        }
        if (profile?.problem_type_suggestion) {
          this.suggestedProblemType = profile.problem_type_suggestion.toLowerCase();
          if (this.suggestedProblemType === 'regression') {
            this.primaryMetric = 'rmse';
          } else {
            this.primaryMetric = 'f1_weighted';
          }
        }
      });
    }
  }

  launch(): void {
    if (!this.selectedDatasetId || !this.experimentName) return;
    this.loading = true;
    this.errorMessage = '';

    this.api.createExperiment({
      dataset_id: this.selectedDatasetId,
      name: this.experimentName,
      target_column: this.targetColumn || undefined,
      primary_metric: this.primaryMetric,
    }).subscribe({
      next: (exp) => {
        this.loading = false;
        this.router.navigate(['/experiments', exp.id]);
      },
      error: (err) => {
        this.loading = false;
        this.errorMessage = err.error?.detail || 'Failed to start experiment.';
      }
    });
  }
}
