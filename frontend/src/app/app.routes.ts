import { Routes } from '@angular/router';
import { authGuard } from './core/auth/auth.guard';
import { ShellComponent } from './layout/shell.component';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    loadComponent: () => import('./landing/landing.component').then(m => m.LandingComponent)
  },
  {
    path: 'auth/login',
    loadComponent: () => import('./auth/login.component').then(m => m.LoginComponent)
  },
  {
    path: 'auth/register',
    loadComponent: () => import('./auth/register.component').then(m => m.RegisterComponent)
  },
  {
    path: '',
    component: ShellComponent,
    canActivate: [authGuard],
    children: [
      {
        path: 'dashboard',
        loadComponent: () => import('./dashboard/dashboard.component').then(m => m.DashboardComponent)
      },
      {
        path: 'datasets',
        loadComponent: () => import('./datasets/dataset-list.component').then(m => m.DatasetListComponent)
      },
      {
        path: 'datasets/:id',
        loadComponent: () => import('./datasets/dataset-detail.component').then(m => m.DatasetDetailComponent)
      },
      {
        path: 'experiments',
        loadComponent: () => import('./dashboard/dashboard.component').then(m => m.DashboardComponent)
      },
      {
        path: 'experiments/new',
        loadComponent: () => import('./experiments/new-experiment.component').then(m => m.NewExperimentComponent)
      },
      {
        path: 'experiments/:id',
        loadComponent: () => import('./experiments/pipeline.component').then(m => m.PipelineComponent)
      },
      {
        path: 'models',
        loadComponent: () => import('./models/comparison.component').then(m => m.ModelComparisonComponent)
      },
      {
        path: 'predictions',
        loadComponent: () => import('./predictions/prediction-form.component').then(m => m.PredictionFormComponent)
      },
      {
        path: 'chat',
        loadComponent: () => import('./chat/chat.component').then(m => m.ChatComponent)
      },
      {
        path: 'reports',
        loadComponent: () => import('./reports/report-viewer.component').then(m => m.ReportViewerComponent)
      }
    ]
  },
  
  { path: '**', redirectTo: '' }
];
