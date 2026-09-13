import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { ApiService } from '../core/services/api.service';
import { Experiment, ExperimentResultModel, ExperimentResults } from '../core/models/types';

interface PipelineStep {
  name: string;
  label: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

@Component({
  selector: 'app-pipeline',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <section *ngIf="experiment" class="analysis-page">
      <header class="page-heading page-heading-with-actions">
        <div>
          <a routerLink="/experiments" class="back-link">Experiments</a>
          <p class="eyebrow">Experiment workspace</p>
          <h1>{{ experiment.name }}</h1>
          <p>Target: {{ experiment.target_column || 'Detecting automatically' }}. The workspace updates as each data science stage is persisted.</p>
        </div>
        <div class="heading-actions">
          <span class="status-pill" [ngClass]="'status-' + experiment.status">{{ experiment.status }}</span>
          <a *ngIf="experiment.status === 'completed'" [routerLink]="['/reports']" [queryParams]="{ experimentId: experiment.id }" class="button button-primary">Open report</a>
          <button *ngIf="experiment.status === 'running'" (click)="cancel()" class="button button-danger">Cancel analysis</button>
        </div>
      </header>

      <section class="surface pipeline-card">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Execution status</p>
            <h2>Analysis pipeline</h2>
          </div>
          <span>{{ completedSteps }} of {{ steps.length }} stages complete</span>
        </div>
        <div class="pipeline-steps">
          <article *ngFor="let step of steps; let index = index" class="pipeline-step" [ngClass]="'step-' + step.status">
            <span class="step-number">{{ index + 1 }}</span>
            <div>
              <strong>{{ step.label }}</strong>
              <small>{{ step.status }}</small>
            </div>
          </article>
        </div>
      </section>

      <section *ngIf="experiment.status === 'completed' && results" class="space-y-6">
        <div class="results-banner">
          <div>
            <p class="eyebrow">Completed analysis</p>
            <h2>{{ champion?.algorithm || 'Model result' }} selected as champion</h2>
            <p>{{ results.models.length }} trained models and {{ results.charts.length }} visual diagnostics are available below.</p>
          </div>
          <div class="banner-metrics">
            <span>Task <strong>{{ experiment.problem_type || 'N/A' }}</strong></span>
            <span>Metric <strong>{{ experiment.primary_metric || 'N/A' }}</strong></span>
          </div>
        </div>

        <div class="summary-grid">
          <article class="surface summary-card">
            <span>Loaded model</span>
            <strong>{{ champion?.algorithm || 'Not available' }}</strong>
            <small>{{ champion?.artifact_available ? champion?.artifact_name : 'No model artifact recorded' }}</small>
          </article>
          <article class="surface summary-card">
            <span>Validation score</span>
            <strong>{{ primaryValue(champion) }}</strong>
            <small>{{ experiment.primary_metric || 'Primary metric' }} on holdout validation data</small>
          </article>
          <article class="surface summary-card">
            <span>Report status</span>
            <strong>{{ results.report.available ? 'Ready' : 'Unavailable' }}</strong>
            <small>{{ results.report.available ? 'PDF contains all generated charts' : 'Review the pipeline log' }}</small>
          </article>
        </div>

        <section class="surface">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Model registry</p>
              <h2>Trained model comparison</h2>
            </div>
            <a [routerLink]="['/models']" [queryParams]="{ experimentId: experiment.id }" class="text-link">Inspect all model details</a>
          </div>
          <div class="model-comparison">
            <article *ngFor="let model of results.models" class="model-result-card" [class.champion-card]="model.is_best">
              <div class="model-card-heading">
                <div>
                  <span class="model-badge" *ngIf="model.is_best">Champion</span>
                  <h3>{{ model.algorithm }}</h3>
                  <p>{{ model.artifact_available ? model.artifact_name : 'Artifact unavailable' }}</p>
                </div>
                <span class="artifact-status" [class.is-ready]="model.artifact_available">{{ model.artifact_available ? 'Loaded' : 'Missing' }}</span>
              </div>
              <div class="metric-row">
                <div *ngFor="let metric of validationMetrics(model)">
                  <span>{{ metric.metric_name }}</span>
                  <strong>{{ metric.metric_value | number:'1.4-4' }}</strong>
                </div>
              </div>
              <div *ngIf="model.hyperparameters && objectEntries(model.hyperparameters).length" class="parameters">
                <span *ngFor="let item of objectEntries(model.hyperparameters)">{{ item.key }}={{ item.value }}</span>
              </div>
            </article>
          </div>
        </section>

        <section class="surface" *ngIf="results.charts.length">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Visual evidence</p>
              <h2>Generated charts</h2>
            </div>
            <span>{{ results.charts.length }} charts</span>
          </div>
          <div class="chart-gallery">
            <figure *ngFor="let chart of results.charts" class="chart-card">
              <div class="chart-frame">
                <img *ngIf="chart.objectUrl" [src]="chart.objectUrl" [alt]="chart.name" />
                <span *ngIf="!chart.objectUrl" class="chart-loading">Loading chart...</span>
              </div>
              <figcaption>{{ chart.name }}</figcaption>
            </figure>
          </div>
        </section>

        <section class="surface" *ngIf="results.preprocessing.length">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Data preparation</p>
              <h2>Applied preprocessing</h2>
            </div>
            <span>{{ results.preprocessing.length }} recorded steps</span>
          </div>
          <div class="preprocessing-list">
            <div *ngFor="let item of results.preprocessing" class="preprocessing-row">
              <span>{{ item.step_order }}</span>
              <strong>{{ item.operation || item.step_type || 'Transformation' }}</strong>
              <p>{{ item.column_name || 'Dataset-wide' }} - {{ item.reason || 'Automated quality improvement' }}</p>
            </div>
          </div>
        </section>
      </section>

      <section class="surface agent-log">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Run history</p>
            <h2>Agent activity</h2>
          </div>
          <span>{{ agentRuns.length }} persisted records</span>
        </div>
        <div *ngIf="agentRuns.length === 0" class="empty-state">Waiting for the first persisted pipeline update...</div>
        <div *ngFor="let run of agentRuns" class="agent-run">
          <span class="agent-status" [ngClass]="'run-' + run.status"></span>
          <div>
            <strong>{{ readableAgent(run.agent_name) }}</strong>
            <p>{{ run.reasoning || run.error_message || 'Stage status was recorded.' }}</p>
          </div>
          <small>{{ run.duration_ms ? (run.duration_ms + ' ms') : run.status }}</small>
        </div>
      </section>
    </section>
  `,
})
export class PipelineComponent implements OnInit, OnDestroy {
  experiment: Experiment | null = null;
  results: ExperimentResults | null = null;
  agentRuns: any[] = [];
  refreshTimer: ReturnType<typeof setInterval> | null = null;

  steps: PipelineStep[] = [
    { name: 'data_understanding', label: 'Data profiling', status: 'pending' },
    { name: 'cleaning', label: 'Data cleaning', status: 'pending' },
    { name: 'eda', label: 'Exploratory analysis', status: 'pending' },
    { name: 'feature_engineering', label: 'Feature engineering', status: 'pending' },
    { name: 'model_selection', label: 'Model selection', status: 'pending' },
    { name: 'training', label: 'Model training', status: 'pending' },
    { name: 'evaluation', label: 'Evaluation', status: 'pending' },
    { name: 'explainability', label: 'Explainability', status: 'pending' },
    { name: 'reporting', label: 'PDF report', status: 'pending' },
  ];

  constructor(private route: ActivatedRoute, private api: ApiService) {}

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id');
    if (!id) return;
    this.refresh(id);
    this.refreshTimer = setInterval(() => this.refresh(id), 3000);
  }

  ngOnDestroy(): void {
    if (this.refreshTimer) clearInterval(this.refreshTimer);
    this.results?.charts.forEach(chart => {
      if (chart.objectUrl) URL.revokeObjectURL(chart.objectUrl);
    });
  }

  get champion(): ExperimentResultModel | null {
    return this.results?.models.find(model => model.is_best) || this.results?.models[0] || null;
  }

  get completedSteps(): number {
    return this.steps.filter(step => step.status === 'completed').length;
  }

  refresh(id: string): void {
    this.api.getExperiment(id).subscribe(experiment => {
      this.experiment = experiment;
      this.api.getAgentRuns(id).subscribe(runs => {
        this.agentRuns = runs;
        this.updateSteps(runs);
      });
      if (experiment.status === 'completed') {
        this.loadResults(id);
        if (this.refreshTimer) {
          clearInterval(this.refreshTimer);
          this.refreshTimer = null;
        }
      }
    });
  }

  loadResults(id: string): void {
    this.api.getExperimentResults(id).subscribe(results => {
      this.results?.charts.forEach(chart => {
        if (chart.objectUrl) URL.revokeObjectURL(chart.objectUrl);
      });
      this.results = results;
      results.charts.forEach(chart => {
        this.api.getChart(chart.url).subscribe({
          next: blob => chart.objectUrl = URL.createObjectURL(blob),
        });
      });
    });
  }

  updateSteps(runs: any[]): void {
    this.steps.forEach(step => {
      const run = runs.filter(item => item.agent_name === step.name).at(-1);
      if (!run) return;
      step.status = run.status === 'failed' ? 'failed' : run.status === 'completed' ? 'completed' : 'running';
    });
  }

  validationMetrics(model: ExperimentResultModel | null): any[] {
    return (model?.metrics || []).filter(metric => metric.split === 'validation').slice(0, 4);
  }

  primaryValue(model: ExperimentResultModel | null): string {
    if (!model || !this.experiment?.primary_metric) return 'N/A';
    const metric = model.metrics.find(item => item.split === 'validation' && item.metric_name === this.experiment?.primary_metric);
    return metric ? metric.metric_value.toFixed(4) : 'N/A';
  }

  objectEntries(value: Record<string, any>): Array<{ key: string; value: any }> {
    return Object.entries(value).map(([key, item]) => ({ key, value: item }));
  }

  readableAgent(name: string): string {
    return name.replace(/_/g, ' ').replace(/\b\w/g, letter => letter.toUpperCase());
  }

  cancel(): void {
    if (!this.experiment) return;
    this.api.cancelExperiment(this.experiment.id).subscribe(() => this.refresh(this.experiment!.id));
  }
}
