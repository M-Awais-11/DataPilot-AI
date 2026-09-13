import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Dataset, DatasetProfile, Experiment, ExperimentResults, TrainedModel, ChatMessage } from '../models/types';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  // Datasets
  uploadDataset(file: File): Observable<Dataset> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<Dataset>(`${this.baseUrl}/datasets/upload`, formData);
  }

  getDatasets(): Observable<Dataset[]> {
    return this.http.get<Dataset[]>(`${this.baseUrl}/datasets`);
  }

  getDataset(id: string): Observable<Dataset> {
    return this.http.get<Dataset>(`${this.baseUrl}/datasets/${id}`);
  }

  getDatasetProfile(id: string): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/datasets/${id}/profile`);
  }

  getDatasetPreview(id: string, rows: number = 100): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/datasets/${id}/preview?n_rows=${rows}`);
  }

  deleteDataset(id: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/datasets/${id}`);
  }

  // Experiments
  createExperiment(data: { dataset_id: string; name: string; target_column?: string; primary_metric?: string }): Observable<Experiment> {
    return this.http.post<Experiment>(`${this.baseUrl}/experiments`, data);
  }

  getExperiments(): Observable<Experiment[]> {
    return this.http.get<Experiment[]>(`${this.baseUrl}/experiments`);
  }

  getExperiment(id: string): Observable<Experiment> {
    return this.http.get<Experiment>(`${this.baseUrl}/experiments/${id}`);
  }

  getAgentRuns(id: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/experiments/${id}/agents`);
  }

  getPreprocessingSteps(id: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/experiments/${id}/preprocessing`);
  }

  getExperimentResults(id: string): Observable<ExperimentResults> {
    return this.http.get<ExperimentResults>(`${this.baseUrl}/experiments/${id}/results`);
  }

  getChart(chartUrl: string): Observable<Blob> {
    return this.http.get(`${this.baseUrl.replace('/api', '')}${chartUrl}`, { responseType: 'blob' });
  }

  cancelExperiment(id: string): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/experiments/${id}/cancel`, {});
  }

  deleteExperiment(id: string): Observable<any> {
    return this.http.delete<any>(`${this.baseUrl}/experiments/${id}`);
  }

  // Server-sent events
  createProgressEventSource(experimentId: string, token: string): EventSource {
    // EventSource does not support custom authorization headers in standard browser API,
    // so token can be sent via query param or cookies if needed, or through direct fetch stream
    return new EventSource(`${this.baseUrl}/experiments/${experimentId}/progress`);
  }

  // Models and registry
  getModels(experimentId?: string): Observable<TrainedModel[]> {
    const url = experimentId ? `${this.baseUrl}/models?experiment_id=${experimentId}` : `${this.baseUrl}/models`;
    return this.http.get<TrainedModel[]>(url);
  }

  getExperimentModels(experimentId: string): Observable<TrainedModel[]> {
    return this.http.get<TrainedModel[]>(`${this.baseUrl}/models/experiment/${experimentId}`);
  }

  getModel(id: string): Observable<TrainedModel> {
    return this.http.get<TrainedModel>(`${this.baseUrl}/models/${id}`);
  }

  promoteModel(id: string): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/models/${id}/promote`, {});
  }

  deleteModel(id: string): Observable<any> {
    return this.http.delete<any>(`${this.baseUrl}/models/${id}`);
  }

  // Predictions
  predict(modelId: string, features: Record<string, any>): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/predict/${modelId}`, { features });
  }

  predictProduction(features: Record<string, any>): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/predict/production`, { features });
  }

  // Reports
  getReportMetadata(experimentId: string): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/reports/${experimentId}`);
  }

  getReportDownloadUrl(experimentId: string): string {
    return `${this.baseUrl}/reports/${experimentId}/download`;
  }

  downloadReport(experimentId: string): Observable<Blob> {
    return this.http.get(`${this.baseUrl}/reports/${experimentId}/download`, { responseType: 'blob' });
  }

  // AI chatbot
  sendChatMessage(experimentId: string, message: string): Observable<ChatMessage> {
    return this.http.post<ChatMessage>(`${this.baseUrl}/chat/${experimentId}`, { message });
  }

  getChatHistory(experimentId: string): Observable<ChatMessage[]> {
    return this.http.get<ChatMessage[]>(`${this.baseUrl}/chat/${experimentId}/history`);
  }
}
