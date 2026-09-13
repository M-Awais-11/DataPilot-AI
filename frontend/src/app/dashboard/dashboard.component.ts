import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { ApiService } from '../core/services/api.service';
import { Dataset, Experiment } from '../core/models/types';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterModule],
  template: `
    <section class="analysis-page dashboard-page">
      <header class="page-heading">
        <div>
          <p class="eyebrow">Workspace overview</p>
          <h1>Build evidence-backed models, faster.</h1>
          <p>Upload a dataset, run a complete analysis, compare every trained model, and retain a PDF report with the full visual evidence.</p>
        </div>
        <a routerLink="/experiments/new" class="button button-primary">Start an analysis</a>
      </header>

      <section class="results-banner dashboard-intro">
        <div>
          <p class="eyebrow">Autonomous workflow</p>
          <h2>From raw CSV to validated model report</h2>
          <p>Profiling, cleaning, exploratory analysis, feature engineering, model evaluation, explainability, and reporting are recorded as one traceable workflow.</p>
        </div>
        <div class="banner-metrics">
          <span>Stages <strong>9</strong></span>
          <span>Charts <strong>All retained</strong></span>
          <span>Output <strong>PDF report</strong></span>
        </div>
      </section>

      <section class="summary-grid dashboard-stats">
        <article class="surface summary-card">
          <span>Datasets</span>
          <strong>{{ datasets.length }}</strong>
          <small>Available for analysis in this workspace</small>
        </article>
        <article class="surface summary-card">
          <span>Experiments</span>
          <strong>{{ experiments.length }}</strong>
          <small>Tracked across all autonomous runs</small>
        </article>
        <article class="surface summary-card">
          <span>Completed reports</span>
          <strong>{{ completedCount }}</strong>
          <small>Results with trained models and deliverables</small>
        </article>
      </section>

      <section class="dashboard-grid">
        <article class="surface dashboard-panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Latest work</p>
              <h2>Experiments</h2>
            </div>
            <a routerLink="/experiments" class="text-link">View all</a>
          </div>
          <div *ngIf="!experiments.length" class="empty-state">No experiments yet. Start by selecting a dataset.</div>

          <div *ngFor="let experiment of experiments.slice(0, 5)"
               class="dashboard-row"
               style="display:flex; align-items:center; gap:8px; padding-right:8px;">
            <a [routerLink]="['/experiments', experiment.id]"
               style="flex:1; display:flex; align-items:center; justify-content:space-between; text-decoration:none; color:inherit;">
              <div>
                <strong>{{ experiment.name }}</strong>
                <small>Target: {{ experiment.target_column || 'Auto-detected' }} / {{ experiment.problem_type || 'Analyzing' }}</small>
              </div>
              <span class="status-pill" [ngClass]="'status-' + experiment.status">{{ experiment.status }}</span>
            </a>
            <button
              (click)="confirmDeleteExperiment(experiment)"
              title="Delete this experiment"
              style="background:none; border:none; cursor:pointer; padding:4px 6px; border-radius:6px; color:#64748b; font-size:15px; line-height:1; flex-shrink:0; transition:color 0.15s;"
              onmouseover="this.style.color='#f87171'"
              onmouseout="this.style.color='#64748b'"
            >🗑️</button>
          </div>
        </article>

        <article class="surface dashboard-panel">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Source data</p>
              <h2>Datasets</h2>
            </div>
            <a routerLink="/datasets" class="text-link">Manage datasets</a>
          </div>
          <div *ngIf="!datasets.length" class="empty-state">No datasets have been uploaded.</div>
          <a *ngFor="let dataset of datasets.slice(0, 5)" [routerLink]="['/datasets', dataset.id]" class="dashboard-row">
            <div>
              <strong>{{ dataset.name }}</strong>
              <small>{{ dataset.row_count ? (dataset.row_count | number) + ' rows' : 'Profiling queued' }} / {{ dataset.col_count || 'Unknown' }} columns</small>
            </div>
            <span class="availability is-ready">{{ dataset.status }}</span>
          </a>
        </article>
      </section>
    </section>

    <!-- Delete Experiment Confirmation Modal -->
    <div *ngIf="experimentToDelete"
         class="fixed inset-0 z-50 flex items-center justify-center p-4"
         style="position:fixed; inset:0; z-index:50; display:flex; align-items:center; justify-content:center; padding:16px; background:rgba(2,6,23,0.85); backdrop-filter:blur(4px);">
      <div style="width:100%; max-width:420px; background:#0f172a; border:1px solid rgba(239,68,68,0.3); border-radius:16px; padding:24px; box-shadow:0 25px 50px rgba(0,0,0,0.5);">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
          <div style="width:40px; height:40px; border-radius:10px; background:rgba(239,68,68,0.1); display:flex; align-items:center; justify-content:center; font-size:20px;">⚠️</div>
          <div>
            <h3 style="color:white; font-size:15px; font-weight:700; margin:0 0 4px;">Delete Experiment?</h3>
            <p style="color:#94a3b8; font-size:11px; margin:0;">Sary models, charts, aur report permanently delete ho jayenge.</p>
          </div>
        </div>

        <div style="background:rgba(2,6,23,0.8); border:1px solid #1e293b; border-radius:8px; padding:12px; font-family:monospace; font-size:11px; margin-bottom:16px;">
          <div style="color:#94a3b8;">Name: <span style="color:white; font-weight:700;">{{ experimentToDelete.name }}</span></div>
          <div style="color:#94a3b8; margin-top:4px;">Status: <span style="color:#cbd5e1;">{{ experimentToDelete.status }}</span></div>
          <div *ngIf="experimentToDelete.target_column" style="color:#94a3b8; margin-top:4px;">Target: <span style="color:#60a5fa;">{{ experimentToDelete.target_column }}</span></div>
        </div>

        <div *ngIf="deleteError" style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:6px; padding:8px 12px; color:#f87171; font-size:11px; margin-bottom:12px;">
          {{ deleteError }}
        </div>

        <div style="display:flex; justify-content:flex-end; gap:10px; padding-top:8px;">
          <button type="button" class="button button-secondary" [disabled]="deleting" (click)="cancelDeleteExperiment()">
            Cancel
          </button>
          <button type="button" class="button button-danger" [disabled]="deleting" (click)="executeDeleteExperiment()">
            {{ deleting ? 'Deleting...' : 'Yes, Delete' }}
          </button>
        </div>
      </div>
    </div>
  `,
})
export class DashboardComponent implements OnInit {
  datasets: Dataset[] = [];
  experiments: Experiment[] = [];
  experimentToDelete: Experiment | null = null;
  deleting = false;
  deleteError = '';

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getDatasets().subscribe(datasets => this.datasets = datasets);
    this.api.getExperiments().subscribe(experiments => this.experiments = experiments);
  }

  get completedCount(): number {
    return this.experiments.filter(e => e.status === 'completed').length;
  }

  confirmDeleteExperiment(experiment: Experiment): void {
    this.experimentToDelete = experiment;
    this.deleteError = '';
  }

  cancelDeleteExperiment(): void {
    this.experimentToDelete = null;
    this.deleteError = '';
  }

  executeDeleteExperiment(): void {
    if (!this.experimentToDelete) return;
    const target = this.experimentToDelete;
    this.deleting = true;
    this.deleteError = '';

    this.api.deleteExperiment(target.id).subscribe({
      next: () => {
        this.experiments = this.experiments.filter(e => e.id !== target.id);
        this.deleting = false;
        this.experimentToDelete = null;
      },
      error: (err) => {
        this.deleting = false;
        this.deleteError = err.error?.detail || 'Delete failed. Please try again.';
      }
    });
  }
}
