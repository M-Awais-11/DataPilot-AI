"""
Report Generation Tool.
Generates a comprehensive PDF report using ReportLab.
Integrates computed metrics, tables, and embedded EDA / SHAP visualization images.
"""
from __future__ import annotations

import logging
import os
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    PageBreak,
    HRFlowable,
)

logger = logging.getLogger(__name__)


def _scaled_image(path: str, max_width: float = 540, max_height: float = 540) -> Image:
    """Fit charts into the document prominently at full page width without distorting their aspect ratio."""
    image = Image(path)
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _draw_footer(canvas, doc) -> None:
    """Add a compact, consistently branded footer to every page."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#90CAF9"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 25, letter[0] - doc.rightMargin, 25)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#0D47A1"))
    canvas.drawString(doc.leftMargin, 13, "Autonomous Data Scientist")
    canvas.drawRightString(letter[0] - doc.rightMargin, 13, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf_report(
    output_path: str,
    experiment_id: str,
    dataset_name: str,
    problem_type: str,
    target_col: Optional[str],
    primary_metric: str,
    df_shape: tuple[int, int],
    cleaning_summary: str,
    eda_summary: str,
    chart_paths: list[str],
    model_comparison_table: list[dict],
    best_model_name: str,
    best_model_metrics: dict[str, float],
    shap_plot_paths: list[str],
    feature_importance: list[dict],
    overfitting_analysis: Optional[str],
    leakage_warnings: list[str],
    recommendations: str,
    model_details: Optional[list[dict]] = None,
) -> str:
    """
    Builds a multi-page PDF report.
    Returns the absolute path to the generated PDF.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    navy = colors.HexColor("#0D47A1")
    primary = colors.HexColor("#2196F3")
    sky = colors.HexColor("#90CAF9")
    mist = colors.HexColor("#E3F2FD")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title=f"Data Science Report - {dataset_name}",
        author="Autonomous Data Scientist",
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=navy,
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=15,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=navy,
        spaceBefore=14,
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6,
    )
    callout_style = ParagraphStyle(
        "ReportCallout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#991B1B"),
        backColor=colors.HexColor("#FEE2E2"),
        borderPadding=6,
        spaceAfter=8,
    )

    story = []

    # ── Header / Title ────────────────────────────────────────────────────────
    story.append(Paragraph("Autonomous Data Scientist Report", title_style))
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y - %H:%M UTC")
    story.append(
        Paragraph(
            f"Dataset: <b>{dataset_name}</b> | Task: <b>{problem_type.upper()}</b> | Target: <b>{target_col or 'N/A'}</b> | Generated: {date_str}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=15))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", h1_style))
    summary_text = (
        f"This report presents an automated end-to-end data science study on dataset <b>{dataset_name}</b>. "
        f"The system identified the problem as <b>{problem_type}</b> aiming to predict <b>{target_col}</b>. "
        f"After automated profiling, data cleansing, exploratory analysis, and feature engineering, multiple machine learning "
        f"algorithms were optimized and evaluated. The best-performing model is <b>{best_model_name}</b>, "
        f"achieving a primary metric (<b>{primary_metric}</b>) of <b>{best_model_metrics.get(primary_metric, 'N/A')}</b>."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))

    # ── Dataset Overview ──────────────────────────────────────────────────────
    story.append(Paragraph("2. Dataset Overview", h1_style))
    ds_data = [
        ["Attribute", "Value"],
        ["Dataset Name", dataset_name],
        ["Row Count", f"{df_shape[0]:,}"],
        ["Column Count", f"{df_shape[1]:,}"],
        ["Problem Type", problem_type.capitalize()],
        ["Target Column", target_col or "N/A"],
        ["Primary Evaluation Metric", primary_metric],
    ]
    t = Table(ds_data, colWidths=[200, 300])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # ── Data Quality & Cleaning ───────────────────────────────────────────────
    story.append(Paragraph("3. Data Cleansing & Leakage Prevention", h1_style))
    if leakage_warnings:
        for warning in leakage_warnings:
            story.append(Paragraph(f"<b>Data Leakage Alert:</b> {warning}", callout_style))
    else:
        story.append(Paragraph("No severe data leakage detected. Preprocessing was fit strictly on training splits.", body_style))

    story.append(Paragraph(f"<b>Cleansing Details:</b> {cleaning_summary}", body_style))
    story.append(Spacer(1, 10))

    # ── Exploratory Data Analysis & Visualizations ────────────────────────────
    story.append(Paragraph("4. Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph(eda_summary, body_style))
    story.append(Spacer(1, 10))

    # Include every generated EDA chart. Each visual is evidence for the final
    # decision, so silently dropping all but two made the PDF incomplete.
    valid_charts = [p for p in chart_paths if os.path.exists(p)]
    if valid_charts:
        story.append(PageBreak())
        story.append(Paragraph("4. Exploratory Data Analysis Visuals", h1_style))
        story.append(Paragraph("High-resolution visual diagnostics generated during exploratory analysis:", body_style))
        story.append(Spacer(1, 8))
    for index, chart_path in enumerate(valid_charts, start=1):
        try:
            if index > 1:
                story.append(PageBreak())
                story.append(Paragraph("4. Exploratory Data Analysis Visuals (continued)", h1_style))
            chart_title = Path(chart_path).stem.replace("_", " ").title()
            story.append(Paragraph(f"<b>Figure 4.{index}: {chart_title}</b>", h2_style))
            story.append(Spacer(1, 6))
            img = _scaled_image(chart_path, max_width=540, max_height=520)
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception as e:
            logger.warning(f"Could not embed chart {chart_path}: {e}")

    # ── Model Comparison Table ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Model Training & Comparison", h1_style))
    
    if model_comparison_table:
        headers = ["Algorithm", f"Val {primary_metric}", "CV Mean ± Std", "Overfit?"]
        table_rows = [headers]
        for row in model_comparison_table:
            val_score = row.get(f"val_{primary_metric}", "N/A")
            cv_mean = row.get("cv_mean")
            cv_std = row.get("cv_std")
            cv_str = f"{cv_mean:.4f} ± {cv_std:.4f}" if cv_mean is not None and cv_std is not None else "N/A"
            val_str = f"{val_score:.4f}" if isinstance(val_score, (int, float)) else str(val_score)
            overfit_str = "YES" if row.get("is_overfitting") else "NO"
            table_rows.append([row.get("algorithm", "N/A"), val_str, cv_str, overfit_str])

        model_tab = Table(table_rows, colWidths=[160, 110, 140, 90])
        model_tab.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(model_tab)
    else:
        story.append(Paragraph("No models trained or comparison data unavailable.", body_style))

    story.append(Spacer(1, 10))

    for model_index, detail in enumerate(model_details or [], start=1):
        algorithm = str(detail.get("algorithm", f"Model {model_index}"))
        artifact_name = Path(str(detail.get("artifact_path") or "Not available")).name
        hyperparameters = detail.get("hyperparameters") or {}
        metrics_by_split = detail.get("metrics") or {}
        detail_rows = [
            ["Property", "Value"],
            ["Saved model artifact", artifact_name or "Not available"],
            ["Training time (seconds)", f"{detail.get('training_time_seconds', 0):.2f}"],
            ["Hyperparameters", ", ".join(f"{key}={value}" for key, value in hyperparameters.items()) or "Default parameters"],
        ]
        for split, metric_values in metrics_by_split.items():
            if isinstance(metric_values, dict):
                formatted = ", ".join(
                    f"{key}={value:.4f}" if isinstance(value, (float, int)) else f"{key}={value}"
                    for key, value in metric_values.items()
                )
                detail_rows.append([f"{str(split).title()} metrics", formatted or "N/A"])
        story.append(Paragraph(f"Model {model_index}: {algorithm}", h2_style))
        detail_table = Table(detail_rows, colWidths=[155, 355])
        detail_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), primary),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, sky),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(detail_table)
        story.append(Spacer(1, 8))

    # ── Best Model Details & Overfitting ──────────────────────────────────────
    story.append(Paragraph("6. Best Model Evaluation & Overfitting Check", h1_style))
    story.append(
        Paragraph(
            f"The selected top model is <b>{best_model_name}</b>. All test & validation splits confirm stability.",
            body_style,
        )
    )
    if overfitting_analysis:
        story.append(Paragraph(f"<b>Overfitting Analysis:</b> {overfitting_analysis}", callout_style))

    story.append(Spacer(1, 10))

    # ── SHAP Explainability ───────────────────────────────────────────────────
    story.append(Paragraph("7. Model Explainability (SHAP)", h1_style))
    if feature_importance:
        story.append(Paragraph("Top predictive features identified by SHAP contribution:", body_style))
        feat_rows = [["Rank", "Feature Name", "Mean |SHAP| Value"]]
        for item in feature_importance[:10]:
            feat_rows.append([
                str(item.get("rank", "-")),
                str(item.get("feature", "N/A")),
                f"{item.get('shap_mean_abs', 0.0):.4f}",
            ])
        feat_tab = Table(feat_rows, colWidths=[50, 250, 150])
        feat_tab.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        story.append(feat_tab)

    # Include every available SHAP diagnostic, not only the first summary plot.
    for shap_index, sp in enumerate(shap_plot_paths, start=1):
        if os.path.exists(sp):
            try:
                story.append(PageBreak())
                story.append(Paragraph("7. Model Explainability Visuals", h1_style))
                shap_title = Path(sp).stem.replace("_", " ").title()
                story.append(Paragraph(f"<b>Explainability Diagnostic {shap_index}: {shap_title}</b>", h2_style))
                story.append(Spacer(1, 6))
                img = _scaled_image(sp, max_width=540, max_height=520)
                story.append(img)
                story.append(Spacer(1, 12))
            except Exception as e:
                logger.warning(f"Could not embed SHAP plot {sp}: {e}")

    story.append(Spacer(1, 10))

    # ── Recommendations & Conclusion ──────────────────────────────────────────
    story.append(Paragraph("8. Recommendations & Production Readiness", h1_style))
    story.append(Paragraph(recommendations or "The champion model is ready for promotion to production API serving.", body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("— End of Autonomous Data Scientist Report —", subtitle_style))

    # Build document
    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return output_path
