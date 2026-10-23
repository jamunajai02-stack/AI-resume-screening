"""HTML report generation for candidate analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.config import DISCLAIMER, TARGET_NOTE


def generate_html_report(
    prediction: dict[str, Any],
    explanation: dict[str, Any],
    job_category: str = "N/A",
) -> str:
    """Generate downloadable HTML candidate analysis report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score = prediction.get("fit_score", 0)
    label = prediction.get("fit_label", "Unknown")
    features = prediction.get("features", {})
    matched = prediction.get("matched_skills", [])
    missing = prediction.get("missing_skills", [])
    positive = explanation.get("positive_factors", [])
    impacts = explanation.get("feature_impacts", [])

    # Label color
    if "Strong" in label:
        label_color = "#27ae60"
    elif "Moderate" in label:
        label_color = "#f39c12"
    else:
        label_color = "#e74c3c"

    matched_html = "".join(f"<span class='badge matched'>{s}</span>" for s in matched) or "<em>None detected</em>"
    missing_html = "".join(f"<span class='badge missing'>{s}</span>" for s in missing) or "<em>None detected</em>"
    positive_html = "".join(f"<li>{f}</li>" for f in positive) or "<li>No strong positive factors identified</li>"

    impacts_rows = ""
    for imp in impacts:
        direction_class = "positive" if imp.get("direction") == "Positive" else "negative"
        impacts_rows += f"""
        <tr>
            <td>{imp.get('feature', '')}</td>
            <td>{imp.get('impact', 0):.4f}</td>
            <td class="{direction_class}">{imp.get('direction', '')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Candidate Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {label_color}; }}
        .label {{ font-size: 24px; color: {label_color}; margin-bottom: 20px; }}
        .metric {{ display: inline-block; background: #ecf0f1; padding: 10px 20px; margin: 5px; border-radius: 8px; }}
        .badge {{ display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 15px; font-size: 13px; }}
        .matched {{ background: #d5f5e3; color: #1e8449; }}
        .missing {{ background: #fadbd8; color: #c0392b; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .disclaimer {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 12px; margin-top: 30px; font-size: 13px; }}
        .footer {{ margin-top: 40px; font-size: 12px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>Candidate Analysis Report</h1>
    <p><strong>Generated:</strong> {now}</p>
    <p><strong>Job Category:</strong> {job_category}</p>

    <h2>Job Fit Prediction</h2>
    <div class="score">{score:.1f}%</div>
    <div class="label">{label}</div>
    <p><em>Predicted job fit based on the trained model.</em></p>

    <h2>Key Metrics</h2>
    <div class="metric">TF-IDF Similarity: {features.get('tfidf_similarity', 0):.3f}</div>
    <div class="metric">Skill Match: {features.get('skill_match_percentage', 0):.1f}%</div>
    <div class="metric">Matched Skills: {features.get('matched_skill_count', 0)}</div>
    <div class="metric">Missing Skills: {features.get('missing_skill_count', 0)}</div>
    <div class="metric">Experience: {features.get('experience_years', 0):.0f} years</div>

    <h2>Matched Skills</h2>
    <div>{matched_html}</div>

    <h2>Skill Gaps</h2>
    <p><em>Missing according to configured skill extraction; not proof candidate lacks the skill.</em></p>
    <div>{missing_html}</div>

    <h2>Positive Factors</h2>
    <ul>{positive_html}</ul>

    <h2>Feature Contributions (SHAP)</h2>
    <table>
        <tr><th>Feature</th><th>Impact</th><th>Direction</th></tr>
        {impacts_rows}
    </table>

    <div class="disclaimer">
        <strong>Disclaimer:</strong> {DISCLAIMER}
        <br><br>
        <strong>Note:</strong> {TARGET_NOTE}
    </div>

    <div class="footer">
        AI-Based Resume Screening and Job Fit Prediction &mdash; Academic Project
    </div>
</body>
</html>"""
    return html
