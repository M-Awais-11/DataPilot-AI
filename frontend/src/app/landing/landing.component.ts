import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { NgFor } from '@angular/common';

@Component({
  selector: 'app-landing',
  standalone: true,
  imports: [RouterModule, NgFor],
  template: `
    <div class="page">

      <!-- Navbar -->
      <nav>
        <a routerLink="/" class="logo">
          <div class="logo-icon">DP</div>
          <span>DataPilot <b>AI</b></span>
        </a>

        <div class="nav-buttons">
          <a routerLink="/auth/login" class="btn login">Login</a>
          <a routerLink="/auth/register" class="btn primary">
            Get Started
          </a>
        </div>
      </nav>

      <!-- Hero -->
      <section class="hero">
        <div class="badge">AI Powered Data Science</div>

        <h1>
          Autonomous Data Science
          <span>Made Simple</span>
        </h1>

        <p>
          Upload your CSV and let DataPilot AI automatically clean your data,
          perform EDA, train machine learning models, explain results,
          and generate a professional report.
        </p>

        <div class="hero-buttons">
          <a routerLink="/auth/register" class="btn primary big">
            Start Free
          </a>

          <a routerLink="/auth/login" class="btn login big">
            Login
          </a>
        </div>
      </section>

      <!-- Stats -->
      <section class="stats">
        <div *ngFor="let stat of stats" class="stat">
          <strong>{{ stat.value }}</strong>
          <span>{{ stat.label }}</span>
        </div>
      </section>

      <!-- Features -->
      <section class="features">
        <div class="section-heading">
          <span>FEATURES</span>
          <h2>Everything You Need</h2>
          <p>
            DataPilot AI automates the complete machine learning workflow.
          </p>
        </div>

        <div class="feature-grid">
          <div *ngFor="let feature of features" class="feature-card">
            <div class="feature-icon">
              {{ feature.icon }}
            </div>

            <h3>{{ feature.title }}</h3>
            <p>{{ feature.desc }}</p>
          </div>
        </div>
      </section>

      <!-- How It Works -->
      <section class="workflow">
        <div class="section-heading">
          <span>WORKFLOW</span>
          <h2>How It Works</h2>
          <p>Just upload your dataset and let AI do the rest.</p>
        </div>

        <div class="steps">
          <div *ngFor="let step of pipeline; let i = index" class="step">
            <div class="step-number">{{ i + 1 }}</div>
            <h3>{{ step.name }}</h3>
            <p>{{ step.desc }}</p>
          </div>
        </div>
      </section>

      <!-- CTA -->
      <section class="cta">
        <h2>Ready to Start?</h2>
        <p>
          Create your free account and start your first data science experiment.
        </p>

        <a routerLink="/auth/register" class="btn primary big">
          Create Free Account
        </a>
      </section>

      <!-- Footer -->
      <footer>
        <strong>DataPilot AI</strong>
        <p>AI-Powered Autonomous Data Science Platform</p>
        <small>© 2026 DataPilot AI. All rights reserved.</small>
      </footer>

    </div>
  `,

  styles: [`
    * {
      box-sizing: border-box;
    }

    :host {
      display: block;
    }

    .page {
      min-height: 100vh;
      background: #f8fafc;
      color: #0f172a;
      font-family: Arial, sans-serif;
    }

    /* Navbar */

    nav {
      height: 70px;
      padding: 0 8%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: white;
      border-bottom: 1px solid #e2e8f0;
    }

    .logo {
      display: flex;
      align-items: center;
      gap: 10px;
      text-decoration: none;
      color: #0f172a;
      font-size: 18px;
      font-weight: bold;
    }

    .logo-icon {
      width: 38px;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #2563eb;
      color: white;
      border-radius: 8px;
      font-size: 14px;
    }

    .logo b {
      color: #2563eb;
    }

    .nav-buttons {
      display: flex;
      gap: 10px;
    }

    /* Buttons */

    .btn {
      padding: 10px 20px;
      border-radius: 7px;
      text-decoration: none;
      font-size: 14px;
      font-weight: 600;
      display: inline-block;
    }

    .primary {
      background: #2563eb;
      color: white;
    }

    .primary:hover {
      background: #1d4ed8;
    }

    .login {
      color: #334155;
      border: 1px solid #cbd5e1;
      background: white;
    }

    .login:hover {
      background: #f1f5f9;
    }

    .big {
      padding: 13px 28px;
      font-size: 15px;
    }

    /* Hero */

    .hero {
      text-align: center;
      max-width: 850px;
      margin: auto;
      padding: 100px 20px 80px;
    }

    .badge {
      display: inline-block;
      padding: 7px 15px;
      border-radius: 20px;
      background: #dbeafe;
      color: #2563eb;
      font-size: 13px;
      font-weight: bold;
      margin-bottom: 20px;
    }

    .hero h1 {
      font-size: 55px;
      line-height: 1.1;
      margin: 0 0 20px;
    }

    .hero h1 span {
      display: block;
      color: #2563eb;
    }

    .hero p {
      max-width: 650px;
      margin: auto;
      color: #64748b;
      font-size: 17px;
      line-height: 1.7;
    }

    .hero-buttons {
      margin-top: 30px;
      display: flex;
      justify-content: center;
      gap: 12px;
    }

    /* Stats */

    .stats {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      background: white;
      border-top: 1px solid #e2e8f0;
      border-bottom: 1px solid #e2e8f0;
    }

    .stat {
      min-width: 150px;
      padding: 25px 35px;
      text-align: center;
      border-right: 1px solid #e2e8f0;
    }

    .stat strong {
      display: block;
      font-size: 25px;
      color: #2563eb;
    }

    .stat span {
      display: block;
      margin-top: 5px;
      font-size: 12px;
      color: #64748b;
    }

    /* Sections */

    .features,
    .workflow {
      max-width: 1100px;
      margin: auto;
      padding: 80px 20px;
    }

    .section-heading {
      text-align: center;
      margin-bottom: 45px;
    }

    .section-heading span {
      color: #2563eb;
      font-size: 12px;
      font-weight: bold;
    }

    .section-heading h2 {
      font-size: 36px;
      margin: 10px 0;
    }

    .section-heading p {
      color: #64748b;
    }

    /* Features */

    .feature-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 20px;
    }

    .feature-card {
      padding: 25px;
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
    }

    .feature-card:hover {
      border-color: #93c5fd;
    }

    .feature-icon {
      width: 45px;
      height: 45px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #dbeafe;
      border-radius: 8px;
      font-size: 22px;
      margin-bottom: 15px;
    }

    .feature-card h3 {
      font-size: 17px;
      margin-bottom: 10px;
    }

    .feature-card p {
      color: #64748b;
      font-size: 14px;
      line-height: 1.6;
    }

    /* Workflow */

    .workflow {
      background: #eff6ff;
      max-width: none;
    }

    .steps {
      max-width: 1000px;
      margin: auto;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 25px;
    }

    .step {
      background: white;
      padding: 25px;
      border-radius: 10px;
      text-align: center;
    }

    .step-number {
      width: 40px;
      height: 40px;
      margin: auto;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #2563eb;
      color: white;
      border-radius: 50%;
      font-weight: bold;
    }

    .step h3 {
      margin: 15px 0 8px;
    }

    .step p {
      color: #64748b;
      font-size: 14px;
    }

    /* CTA */

    .cta {
      text-align: center;
      padding: 90px 20px;
      background: white;
    }

    .cta h2 {
      font-size: 36px;
      margin-bottom: 10px;
    }

    .cta p {
      color: #64748b;
      margin-bottom: 30px;
    }

    /* Footer */

    footer {
      text-align: center;
      padding: 35px 20px;
      background: #0f172a;
      color: white;
    }

    footer p {
      color: #94a3b8;
      font-size: 13px;
    }

    footer small {
      color: #64748b;
    }

    /* Responsive */

    @media (max-width: 768px) {

      nav {
        padding: 0 20px;
      }

      .hero h1 {
        font-size: 40px;
      }

      .feature-grid,
      .steps {
        grid-template-columns: 1fr;
      }

      .stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
      }

      .stat {
        border-bottom: 1px solid #e2e8f0;
      }

      .hero-buttons {
        flex-direction: column;
        align-items: center;
      }
    }
  `]
})
export class LandingComponent {

  stats = [
    { value: '10+', label: 'AI Agents' },
    { value: 'AutoML', label: 'Model Training' },
    { value: 'SHAP', label: 'Explainability' },
    { value: 'PDF', label: 'Reports' }
  ];

  features = [
    {
      icon: '📊',
      title: 'Automatic EDA',
      desc: 'Analyze your dataset automatically with statistics, distributions and correlations.'
    },
    {
      icon: '🧹',
      title: 'Data Cleaning',
      desc: 'Automatically handle missing values, duplicates, outliers and categorical data.'
    },
    {
      icon: '🤖',
      title: 'Machine Learning',
      desc: 'Train and compare multiple machine learning models automatically.'
    },
    {
      icon: '⚙️',
      title: 'Hyperparameter Tuning',
      desc: 'Optimize model parameters automatically to achieve better performance.'
    },
    {
      icon: '🔍',
      title: 'Model Explainability',
      desc: 'Understand model predictions using SHAP and AI-generated explanations.'
    },
    {
      icon: '📄',
      title: 'PDF Reports',
      desc: 'Generate professional reports containing results, charts and recommendations.'
    }
  ];

  pipeline = [
    {
      name: 'Upload',
      desc: 'Upload your CSV dataset.'
    },
    {
      name: 'Clean',
      desc: 'AI cleans your data.'
    },
    {
      name: 'EDA',
      desc: 'Analyze your dataset.'
    },
    {
      name: 'Features',
      desc: 'Select useful features.'
    },
    {
      name: 'Training',
      desc: 'Train ML models.'
    },
    {
      name: 'Evaluation',
      desc: 'Compare model results.'
    },
    {
      name: 'Explain',
      desc: 'Understand predictions.'
    },
    {
      name: 'Report',
      desc: 'Generate final report.'
    },
    {
      name: 'Chat',
      desc: 'Ask AI questions.'
    }
  ];
}