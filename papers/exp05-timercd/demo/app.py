import io
import zipfile
from pathlib import Path
from typing import List, Tuple, Literal, Optional
from evaluation.metrics import get_metrics
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from model_wrapper import run_Time_RCD

REPO_ID = "thu-sail-lab/Time-RCD"

CHECKPOINT_FILES = [
    "checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth",
    "checkpoints/dataset_10_20.pth",
    "checkpoints/full_mask_10_20.pth",
    "checkpoints/dataset_15_56.pth",
    "checkpoints/full_mask_15_56.pth",
]


def ensure_checkpoints() -> None:
    """Ensure that the required checkpoint files are present locally."""
    missing = [path for path in CHECKPOINT_FILES if not Path(path).exists()]
    if not missing:
        return

    try:
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="checkpoints.zip",
            repo_type="model",
            cache_dir=".cache/hf",
        )
    except HfHubHTTPError:
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="checkpoints.zip",
            repo_type="dataset",
            cache_dir=".cache/hf",
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATASET_DIR = BASE_DIR / "sample_datasets"

LabelSource = Literal["same_file", "separate_file", "none"]
LABEL_COLUMN_CANDIDATES = ("label", "labels")
LABEL_SOURCE_CHOICES = {
    "Value + label in same file": "same_file",
    "Labels in separate file": "separate_file",
    "No labels provided": "none",
}
SAMPLE_FILES: dict[str, dict[str, object]] = {
    "Sample: Univariate SED Medical": {
        "path": SAMPLE_DATASET_DIR / "235_SED_id_2_Medical_tr_2499_1st_3840.csv",
        "is_multivariate": False,
    },
    "Sample: Univariate UCR Medical": {
        "path": SAMPLE_DATASET_DIR / "353_UCR_id_51_Medical_tr_1875_1st_3198.csv",
        "is_multivariate": False,
    },
    "Sample: Univariate Yahoo WebService": {
        "path": SAMPLE_DATASET_DIR / "686_YAHOO_id_136_WebService_tr_500_1st_755.csv",
        "is_multivariate": False,
    },
    # "Sample: Multivariate MSL Sensor": {
    #     "path": SAMPLE_DATASET_DIR / "003_MSL_id_2_Sensor_tr_883_1st_1238.csv",
    #     "is_multivariate": True,
    # },
}


def _resolve_path(file_obj) -> Path:
    """Extract a pathlib.Path from the gradio file object."""
    if file_obj is None:
        raise ValueError("File object is None.")

    if isinstance(file_obj, Path):
        return file_obj

    if isinstance(file_obj, str):
        path = Path(file_obj)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        return path

    # Gradio may pass dictionaries or objects with a 'name' attribute.
    if isinstance(file_obj, dict) and "name" in file_obj:
        return _resolve_path(file_obj["name"])

    name = getattr(file_obj, "name", None)
    if not name:
        raise ValueError("Unable to resolve uploaded file path.")
    return _resolve_path(name)


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from supported file types."""
    if not path.exists():
        raise ValueError(f"File not found: {path}. If this is a bundled sample, ensure it exists under {SAMPLE_DATASET_DIR}.")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if not isinstance(data, np.ndarray):
            raise ValueError("Loaded .npy data is not a numpy array.")
        return pd.DataFrame(data)

    if suffix not in {".csv", ".txt"}:
        raise ValueError("Unsupported file type. Please upload a .csv, .txt, or .npy file.")

    return pd.read_csv(path)


def _extract_label_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Split a label column from dataframe if one of the candidate names exists."""
    lower_to_original = {col.lower(): col for col in df.columns}
    label_col = None
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            label_col = lower_to_original[candidate]
            break

    if label_col is None:
        return df, None

    label_series = pd.to_numeric(df[label_col], errors="raise")
    feature_df = df.drop(columns=[label_col])
    return feature_df, label_series


def _load_label_series(file_obj) -> pd.Series:
    """Load labels from a dedicated upload."""
    path = _resolve_path(file_obj)
    df = _load_dataframe(path)
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        raise ValueError("Uploaded label file does not contain numeric columns.")

    lower_to_original = {col.lower(): col for col in numeric_df.columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            column = lower_to_original[candidate]
            return pd.to_numeric(numeric_df[column], errors="raise").rename("label")

    if numeric_df.shape[1] > 1:
        raise ValueError(
            "Label file must contain exactly one numeric column or include a column named 'label'."
        )

    series = pd.to_numeric(numeric_df.iloc[:, 0], errors="raise").rename("label")
    return series


def load_timeseries(
    value_file,
    feature_columns: List[str] | None,
    label_source: LabelSource,
    label_file=None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.Series]]:
    """Load the uploaded value file, optional label file, and return features/labels."""
    value_path = _resolve_path(value_file)
    raw_df = _load_dataframe(value_path)
    feature_df = raw_df.select_dtypes(include=np.number)

    if feature_df.empty:
        raise ValueError("No numeric columns detected. Ensure your value file contains numeric values.")

    label_series: Optional[pd.Series] = None
    feature_df, embedded_label = _extract_label_column(feature_df)

    if label_source == "same_file":
        if embedded_label is None:
            raise ValueError(
                "Label column not found in the uploaded file. Expected a column named 'label'."
            )
        label_series = embedded_label
    elif label_source == "separate_file":
        if label_file is None:
            raise ValueError("Please upload a label file or switch the label source option.")
        label_series = _load_label_series(label_file)
    elif label_source == "none":
        label_series = None
    else:
        raise ValueError(f"Unsupported label source option: {label_source}")

    if feature_columns:
        missing = [col for col in feature_columns if col not in feature_df.columns]
        if missing:
            raise ValueError(f"Selected columns not found in the value file: {', '.join(missing)}")
        feature_df = feature_df[feature_columns]

    feature_df = feature_df.reset_index(drop=True)

    if label_series is not None:
        label_series = label_series.reset_index(drop=True)
        if len(label_series) != len(feature_df):
            min_length = min(len(label_series), len(feature_df))
            label_series = label_series.iloc[:min_length].reset_index(drop=True)
            feature_df = feature_df.iloc[:min_length, :].reset_index(drop=True)

    array = feature_df.to_numpy(dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    return feature_df, array, label_series


def _metrics_to_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame({"Metric": [], "Value": []})
    return pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": [round(float(value), 4) for value in metrics.values()],
        }
    )


def infer(
    file_obj,
    is_multivariate: bool,
    window_size: int,
    batch_size: int,
    multi_size: str,
    feature_columns: List[str],
    label_source: LabelSource,
    label_file,
) -> Tuple[str, pd.DataFrame, plt.Figure, pd.DataFrame]:
    """Run Time-RCD inference and produce outputs for the Gradio UI."""
    ensure_checkpoints()
    numeric_df, array, labels = load_timeseries(
        file_obj, feature_columns or None, label_source=label_source, label_file=label_file
    )

    num_features = array.shape[1] if array.ndim > 1 else 1
    if is_multivariate and num_features == 1:
        raise ValueError(
            "Dataset check: only one feature column found, so please switch the Data type to 'Univariate' or upload a multivariate file with multiple feature columns."
        )
    if not is_multivariate and num_features > 1:
        raise ValueError(
            "Dataset check: multiple feature columns detected, so please switch the Data type to 'Multivariate' or provide a univariate file with a single feature column."
        )

    kwargs = {
        "Multi": is_multivariate,
        "win_size": window_size,
        "batch_size": batch_size,
        "random_mask": "random_mask",
        "size": multi_size,
        "device": "cpu",
    }

    scores, logits = run_Time_RCD(array, **kwargs)
    score_vector = np.asarray(scores).reshape(-1)
    logit_vector = np.asarray(logits).reshape(-1)

    valid_length = min(len(score_vector), len(numeric_df))
    if labels is not None:
        valid_length = min(valid_length, len(labels))

    result_df = numeric_df.iloc[:valid_length, :].copy()
    score_series = pd.Series(score_vector[:valid_length], index=result_df.index, name="anomaly_score")
    logit_series = pd.Series(logit_vector[:valid_length], index=result_df.index, name="anomaly_logit")
    result_df["anomaly_score"] = score_series
    result_df["anomaly_logit"] = logit_series

    metrics_df: pd.DataFrame
    if labels is not None:
        label_series = labels.iloc[:valid_length]
        result_df["label"] = label_series.to_numpy()
        metrics = get_metrics(score_series.to_numpy(), label_series.to_numpy())
        metrics_df = _metrics_to_dataframe(metrics)
    else:
        metrics_df = pd.DataFrame({"Metric": ["Info"], "Value": ["Labels not provided; metrics skipped."]})

    top_indices = score_series.nlargest(5).index.tolist()
    highlight_message = (
        "Top anomaly indices (by score): " + ", ".join(str(idx) for idx in top_indices)
        if len(top_indices) > 0
        else "No anomalies detected."
    )
    if labels is None:
        highlight_message += " Metrics skipped due to missing labels."

    figure = build_plot(result_df)

    return highlight_message, result_df, figure, metrics_df


def build_plot(result_df: pd.DataFrame) -> plt.Figure:
    """Create a matplotlib plot of the first feature vs. anomaly score."""
    fig, ax_primary = plt.subplots(
        figsize=(12, 4),    # wider canvas
        dpi=200,            # higher resolution
        constrained_layout=True
    )
    index = result_df.index
    feature_cols = [
        col for col in result_df.columns if col not in {"anomaly_score", "anomaly_logit", "label"}
    ]

    primary_col = feature_cols[0]
    ax_primary.plot(
        index,
        result_df[primary_col],
        label=f"{primary_col}",
        color="#1f77b4",
        linewidth=1.0,
    )

    if "label" in result_df.columns:
        anomalies = result_df[result_df["label"] > 0]
        if not anomalies.empty:
            ax_primary.scatter(
                anomalies.index,
                anomalies[primary_col],
                label="Label = 1",
                color="#ff7f0e",
                marker="o",
                s=30,
                alpha=0.85,
            )

    ax_primary.set_xlabel("Index")
    ax_primary.set_ylabel("Value")
    ax_primary.grid(alpha=0.2)

    ax_secondary = ax_primary.twinx()
    ax_secondary.plot(
        index,
        result_df["anomaly_score"],
        label="Anomaly Score",
        color="#d62728",
        linewidth=1.0,
    )
    ax_secondary.set_ylabel("Anomaly Score")

    handles_primary, labels_primary = ax_primary.get_legend_handles_labels()
    handles_secondary, labels_secondary = ax_secondary.get_legend_handles_labels()
    ax_primary.legend(
        handles_primary + handles_secondary,
        labels_primary + labels_secondary,
        loc="upper right",
    )

    fig.tight_layout()
    return fig


def build_interface() -> gr.Blocks:
    """Define the Gradio UI."""
    with gr.Blocks(title="Time-RCD Zero-Shot Anomaly Detection") as demo:
        gr.Markdown(
            "# Time-RCD Zero-Shot Anomaly Detection\n"
            "Start with one of the bundled datasets or upload your own time series to run zero-shot anomaly detection."
        )

        bundled_choices = list(SAMPLE_FILES.keys())
        default_choice = bundled_choices[0] if bundled_choices else "Upload my own"

        data_selector = gr.Radio(
            choices=bundled_choices + ["Upload my own"],
            value=default_choice,
            label="Choose dataset",
        )

        with gr.Row():
            file_input = gr.File(
                label="Upload time series file (.csv, .txt, .npy)",
                file_types=[".csv", ".txt", ".npy"],
                visible=default_choice == "Upload my own",
            )
            label_source = gr.Radio(
                choices=list(LABEL_SOURCE_CHOICES.keys()),
                value="Value + label in same file",
                label="Label source",
            )

        with gr.Row():
            label_file_input = gr.File(
                label="Upload label file (.csv, .txt, .npy)",
                file_types=[".csv", ".txt", ".npy"],
                visible=False,
            )
            column_selector = gr.Textbox(
                label="Columns to use (comma-separated, optional)",
                placeholder="e.g. value,feature_1,feature_2",
            )

        gr.Markdown(
            "Bundled datasets live in the Downloads folder and include labels unless noted. "
            "Select \"Upload my own\" to provide a custom file."
        )

        with gr.Row():
            multivariate = gr.Radio(
                choices=["Univariate", "Multivariate"],
                value=(
                    "Multivariate"
                    if bundled_choices and SAMPLE_FILES[default_choice]["is_multivariate"]
                    else "Univariate"
                ),
                label="Data type",
            )
            window_size_in = gr.Slider(
                minimum=128,
                maximum=20000,
                value=15000,
                step=128,
                label="Window size",
            )
            batch_size_in = gr.Slider(
                minimum=1,
                maximum=128,
                value=16,
                step=1,
                label="Batch size",
            )

        with gr.Row():
            multi_size_in = gr.Radio(
                choices=["full", "small"],
                value="full",
                label="Multivariate model size",
            )

        run_button = gr.Button("Run Inference", variant="primary")

        result_message = gr.Textbox(label="Summary", interactive=False)
        result_dataframe = gr.DataFrame(label="Anomaly Scores", interactive=False)
        plot_output = gr.Plot(label="Series vs. Anomaly Score")
        metrics_output = gr.DataFrame(label="Metrics", interactive=False)
        def _submit(
            data_choice,
            file_obj,
            label_source_choice,
            label_file_obj,
            multivariate_choice,
            win,
            batch,
            size,
            columns_text,
        ):
            use_sample = data_choice != "Upload my own"
            if use_sample:
                sample_entry = SAMPLE_FILES[data_choice]
                value_obj = sample_entry["path"]
            else:
                value_obj = file_obj

            if value_obj is None:
                raise gr.Error("Please upload a time series file or choose a sample.")

            feature_columns = [col.strip() for col in columns_text.split(",") if col.strip()] if columns_text else []
            is_multi = multivariate_choice == "Multivariate"
            resolved_label_source = LABEL_SOURCE_CHOICES[label_source_choice]
            if resolved_label_source == "separate_file" and label_file_obj is None:
                raise gr.Error("Please upload a label file or change the label source option.")
            summary, df, fig, metrics = infer(
                file_obj=value_obj,
                is_multivariate=is_multi,
                window_size=int(win),
                batch_size=int(batch),
                multi_size=size,
                feature_columns=feature_columns,
                label_source=resolved_label_source,
                label_file=label_file_obj,
            )
            return summary, df, fig, metrics

        def _toggle_label_file(option):
            return gr.update(visible=option == "Labels in separate file")

        def _handle_dataset_choice(choice):
            show_upload = choice == "Upload my own"
            if choice == "Upload my own":
                multi_update = gr.update()
            else:
                expected_multi = SAMPLE_FILES[choice]["is_multivariate"]
                multi_update = gr.update(value="Multivariate" if expected_multi else "Univariate")
            return gr.update(visible=show_upload), multi_update

        label_source.change(fn=_toggle_label_file, inputs=label_source, outputs=label_file_input)
        data_selector.change(fn=_handle_dataset_choice, inputs=data_selector, outputs=[file_input, multivariate])

        run_button.click(
            fn=_submit,
            inputs=[
                data_selector,
                file_input,
                label_source,
                label_file_input,
                multivariate,
                window_size_in,
                batch_size_in,
                multi_size_in,
                column_selector,
            ],
            outputs=[result_message, result_dataframe, plot_output, metrics_output],
        )

    return demo


demo = build_interface()

if __name__ == "__main__":
    demo.launch()

