
import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

# Optional PIL support for dynamic responsive scaling
HAS_PIL = False
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    DECISION_TREE_MODEL, KMEANS_MODEL, LINEAR_REG_MODEL,
    RESULTS_DIR, ALL_FEATURES, SOIL_FEATURES,
    ensure_dirs, setup_logger
)

logger = setup_logger("gui")

# ──────────────────────────────────────────────
# COLOR PALETTE DEFINITIONS (Premium Dark Theme)
# ──────────────────────────────────────────────
BG_MAIN = "#0f0f12"         # Deep Space Dark
BG_CARD = "#17171c"         # Sleek Obsidian Slate
BG_INPUT = "#1e1e24"        # Dark entry field background
ACCENT_CYAN = "#00f5ff"     # Neon Cyan (Primary Highlight)
ACCENT_PURPLE = "#a29bfe"   # Soft Purple (Secondary Highlight)
ACCENT_GREEN = "#2ecc71"    # Emerald Green (Status/Yield)
ACCENT_ORANGE = "#ff9f43"   # warm Amber (Warning/Alert)
TEXT_WHITE = "#ffffff"      # Primary Text
TEXT_SILVER = "#a0a0b2"     # Cool Secondary Text
TEXT_DARK = "#0f0f12"       # Contrast Text
BORDER_COLOR = "#2d2d35"    # Subtle borders
BORDER_ACTIVE = "#00f5ff"    # Glowing border on focus

# Pre-rendered plot image paths
PLOT_FEATURE_IMPORTANCE = os.path.join(RESULTS_DIR, "feature_importance.png")
PLOT_CLUSTER_SCATTER = os.path.join(RESULTS_DIR, "cluster_scatter.png")
PLOT_RESIDUAL = os.path.join(RESULTS_DIR, "residual_analysis.png")


class ResponsivePlotCanvas(tk.Canvas):
    """A canvas that automatically fits and scales a plot image when resized."""
    def __init__(self, parent, image_path, fallback_text, **kwargs):
        super().__init__(parent, bg=BG_CARD, highlightthickness=0, **kwargs)
        self.image_path = image_path
        self.fallback_text = fallback_text
        self.pil_image = None
        self.tk_image = None
        
        # Bind resize event
        self.bind("<Configure>", self.on_resize)
        
        # Load image if it exists
        if os.path.exists(self.image_path):
            if HAS_PIL:
                try:
                    self.pil_image = Image.open(self.image_path)
                except Exception as e:
                    logger.error(f"Error opening PIL image {self.image_path}: {e}")
            else:
                try:
                    # Native PhotoImage fallback
                    self.tk_image = tk.PhotoImage(file=self.image_path)
                except Exception as e:
                    logger.error(f"Error opening native photo {self.image_path}: {e}")
        
        if not os.path.exists(self.image_path):
            self.create_text_fallback()

    def create_text_fallback(self):
        self.delete("all")
        self.create_text(
            self.winfo_width() / 2 or 200,
            self.winfo_height() / 2 or 100,
            text=f"{self.fallback_text}\n(Run models.py to generate plots)",
            fill=TEXT_SILVER,
            font=("Helvetica", 11),
            justify="center"
        )

    def on_resize(self, event):
        """Called dynamically on window/canvas resize."""
        if not os.path.exists(self.image_path):
            self.create_text_fallback()
            return

        w, h = event.width, event.height
        if w < 10 or h < 10:
            return

        self.delete("all")

        if HAS_PIL and self.pil_image:
            try:
                # Aspect ratio scaling
                img_w, img_h = self.pil_image.size
                ratio = min(w / img_w, h / img_h)
                new_w = int(img_w * ratio)
                new_h = int(img_h * ratio)

                # Avoid scaling to 0
                new_w = max(1, new_w)
                new_h = max(1, new_h)

                resized_img = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                self.tk_image = ImageTk.PhotoImage(resized_img)
                
                # Center inside canvas
                self.create_image(w / 2, h / 2, image=self.tk_image, anchor="center")
            except Exception as e:
                logger.error(f"Resize failed: {e}")
        elif self.tk_image:
            # Fallback static centering
            self.create_image(w / 2, h / 2, image=self.tk_image, anchor="center")
        else:
            self.create_text_fallback()


class SmartAgriGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Smart Agriculture Decision Support System (DSS)")
        self.geometry("1920x1200")
        self.configure(bg=BG_MAIN)
        self.minsize(1920, 1200)

        ensure_dirs()
        self.load_artifacts()
        self.create_styles()
        self.build_ui()

    def load_artifacts(self):
        """Load all serialized ML models and metrics."""
        self.models_loaded = False
        try:
            if not os.path.exists(DECISION_TREE_MODEL):
                raise FileNotFoundError(f"Decision Tree model not found at {DECISION_TREE_MODEL}")
            self.dt_clf = joblib.load(DECISION_TREE_MODEL)

            if not os.path.exists(KMEANS_MODEL):
                raise FileNotFoundError(f"K-Means bundle not found at {KMEANS_MODEL}")
            self.kmeans_bundle = joblib.load(KMEANS_MODEL)
            self.kmeans = self.kmeans_bundle["kmeans"]
            self.soil_scaler = self.kmeans_bundle["scaler"]

            if not os.path.exists(LINEAR_REG_MODEL):
                raise FileNotFoundError(f"Linear Regression model not found at {LINEAR_REG_MODEL}")
            self.lr_model = joblib.load(LINEAR_REG_MODEL)

            self.load_metrics_data()
            self.models_loaded = True
            logger.info("All ML models and metadata loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            messagebox.showerror(
                "Model Loading Failure",
                f"Could not load ML models.\nRun preprocessing.py then models.py first!\n\nError: {e}"
            )
            self.dt_clf = None
            self.kmeans = None
            self.lr_model = None

    def load_metrics_data(self):
        """Load metric json files for displaying precision and bounds."""
        self.residual_std = 5.13
        self.agronomic_guidance = {}

        km_metrics_path = os.path.join(RESULTS_DIR, "kmeans_metrics.json")
        lr_metrics_path = os.path.join(RESULTS_DIR, "linear_regression_metrics.json")

        if os.path.exists(km_metrics_path):
            with open(km_metrics_path, "r") as f:
                km_data = json.load(f)
                self.agronomic_guidance = km_data.get("agronomic_guidance", {})

        if os.path.exists(lr_metrics_path):
            with open(lr_metrics_path, "r") as f:
                lr_data = json.load(f)
                self.residual_std = lr_data.get("residual_stats", {}).get("std", 5.13)

    def create_styles(self):
        """Define custom styles for the Dark UI Theme."""
        style = ttk.Style()
        style.theme_use("default")

        style.configure("TFrame", background=BG_MAIN)
        style.configure("Card.TFrame", background=BG_CARD, relief="flat")

        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=BG_CARD, foreground=TEXT_SILVER,
                        padding=[15, 6], font=("Helvetica", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT_CYAN)],
                  foreground=[("selected", TEXT_DARK)])

        style.configure("TLabel", background=BG_MAIN, foreground=TEXT_WHITE, font=("Helvetica", 10))
        style.configure("Title.TLabel", background=BG_MAIN, foreground=ACCENT_CYAN, font=("Helvetica", 16, "bold"))
        style.configure("Header.TLabel", background=BG_CARD, foreground=TEXT_WHITE, font=("Helvetica", 12, "bold"))
        style.configure("ResultTitle.TLabel", background=BG_CARD, foreground=ACCENT_PURPLE, font=("Helvetica", 10, "bold"))
        style.configure("ResultValue.TLabel", background=BG_CARD, foreground=TEXT_WHITE, font=("Helvetica", 14, "bold"))

        style.configure("TEntry", fieldbackground=BG_INPUT, foreground=TEXT_WHITE, insertcolor=TEXT_WHITE, borderwidth=0)

        style.configure("TButton", background=ACCENT_CYAN, foreground=TEXT_DARK,
                        font=("Helvetica", 10, "bold"), padding=[12, 6], borderwidth=0)
        style.map("TButton",
                  background=[("active", ACCENT_PURPLE)],
                  foreground=[("active", TEXT_WHITE)])

        style.configure("Secondary.TButton", background=BG_CARD, foreground=TEXT_SILVER,
                        font=("Helvetica", 9, "bold"), padding=[10, 5], borderwidth=0)
        style.map("Secondary.TButton",
                  background=[("active", ACCENT_CYAN)],
                  foreground=[("active", TEXT_DARK)])

    def build_ui(self):
        """Construct the dashboard grids and panels."""
        self.columnconfigure(0, weight=3)  # Left panel (Inputs)
        self.columnconfigure(1, weight=5)  # Right panel (Outputs & Charts)
        self.rowconfigure(0, weight=0)     # Header
        self.rowconfigure(1, weight=1)     # Content rows

        # ── HEADER BAR ──
        header_frame = ttk.Frame(self, padding=12)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        title_lbl = ttk.Label(header_frame, text="AGRICULTURAL DECISION SUPPORT SYSTEM (DSS)", style="Title.TLabel")
        title_lbl.pack(side="left")

        status_text = "Algorithmic Core Connected" if self.models_loaded else "Algorithmic Connection Offline"
        status_color = ACCENT_GREEN if self.models_loaded else ACCENT_ORANGE
        status_lbl = ttk.Label(header_frame, text=status_text,
                               foreground=status_color, font=("Helvetica", 10, "bold"))
        status_lbl.pack(side="right")

        # ── LEFT PANEL: Parameter Inputs ──
        input_container = ttk.Frame(self, padding=10)
        input_container.grid(row=1, column=0, sticky="nsew")
        input_container.rowconfigure(0, weight=1)
        input_container.columnconfigure(0, weight=1)

        input_card = ttk.Frame(input_container, style="Card.TFrame", padding=15)
        input_card.grid(row=0, column=0, sticky="nsew")
        input_card.columnconfigure(0, weight=1)
        input_card.columnconfigure(1, weight=1)

        input_header = ttk.Label(input_card, text="Soil & Climatic Inputs", style="Header.TLabel")
        input_header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

        self.input_vars = {}
        features_info = [
            ("N", "Nitrogen (N) - kg/ha", "90", "Range: 0 to 140"),
            ("P", "Phosphorus (P) - kg/ha", "42", "Range: 5 to 145"),
            ("K", "Potassium (K) - kg/ha", "43", "Range: 5 to 205"),
            ("ph", "Soil pH Level", "6.5", "Range: 3.5 to 9.9"),
            ("temperature", "Temperature (°C)", "20.8", "Range: 10 to 45"),
            ("humidity", "Relative Humidity (%)", "82.0", "Range: 15 to 100"),
            ("rainfall", "Precipitation / Rain (mm)", "202.9", "Range: 20 to 300")
        ]

        row_idx = 1
        for key, name, default, limit in features_info:
            lbl = ttk.Label(input_card, text=name, background=BG_CARD, font=("Helvetica", 10, "bold"))
            lbl.grid(row=row_idx, column=0, sticky="w", pady=(8, 2))

            # Beautiful styled wrapper frame for entries to act as a glowing border
            entry_wrapper = tk.Frame(input_card, bg=BORDER_COLOR, bd=1)
            entry_wrapper.grid(row=row_idx+1, column=0, columnspan=2, sticky="ew", pady=(0, 6))

            var = tk.StringVar(value=default)
            self.input_vars[key] = var

            entry = tk.Entry(
                entry_wrapper, textvariable=var, font=("Helvetica", 10),
                bg=BG_INPUT, fg=TEXT_WHITE, insertbackground=TEXT_WHITE,
                relief="flat", bd=4
            )
            entry.pack(fill="x", expand=True)

            # Bind interactive glowing animations
            entry.bind("<FocusIn>", lambda e, w=entry_wrapper: w.config(bg=BORDER_ACTIVE))
            entry.bind("<FocusOut>", lambda e, w=entry_wrapper: w.config(bg=BORDER_COLOR))

            # Helper bound details
            bound_lbl = ttk.Label(input_card, text=limit, background=BG_CARD,
                                  foreground=TEXT_SILVER, font=("Helvetica", 8))
            bound_lbl.grid(row=row_idx, column=1, sticky="e", pady=(8, 2))

            row_idx += 2

        # Control Panel Buttons
        btn_frame = ttk.Frame(input_card, style="Card.TFrame")
        btn_frame.grid(row=row_idx, column=0, columnspan=2, pady=(15, 0), sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        load_btn = ttk.Button(btn_frame, text="Load Preset (Rice)", command=self.load_preset, style="Secondary.TButton")
        load_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        run_btn = ttk.Button(btn_frame, text="Run Analysis", command=self.run_predictions)
        run_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # ── RIGHT PANEL: Integrated Results & Visualizations ──
        right_container = ttk.Frame(self, padding=10)
        right_container.grid(row=1, column=1, sticky="nsew")
        right_container.columnconfigure(0, weight=1)
        right_container.rowconfigure(0, weight=3) # Outputs
        right_container.rowconfigure(1, weight=4) # Visualizations Notebook

        # Outputs Dashboard Card
        output_card = ttk.Frame(right_container, style="Card.TFrame", padding=15)
        output_card.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        output_header = ttk.Label(output_card, text="Agronomic Analysis & Predictions", style="Header.TLabel")
        output_header.pack(anchor="w", pady=(0, 15))

        # Model result grids inside output card
        results_grid = ttk.Frame(output_card, style="Card.TFrame")
        results_grid.pack(fill="both", expand=True)
        results_grid.columnconfigure(0, weight=1)
        results_grid.columnconfigure(1, weight=1)
        results_grid.rowconfigure(0, weight=1)
        results_grid.rowconfigure(1, weight=1)

        # 1. Recommended crop type card
        crop_frame = tk.Frame(results_grid, bg=BG_MAIN, bd=1, relief="solid", highlightthickness=0, colormap="new")
        crop_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        crop_lbl_title = tk.Label(crop_frame, text="RECOMMENDED CROP TYPE", bg=BG_MAIN, fg=ACCENT_CYAN, font=("Helvetica", 9, "bold"))
        crop_lbl_title.pack(anchor="w", padx=12, pady=(10, 2))
        self.crop_res_val = tk.Label(crop_frame, text="Waiting for analysis...", bg=BG_MAIN, fg=TEXT_WHITE, font=("Helvetica", 14, "bold"))
        self.crop_res_val.pack(anchor="w", padx=12, pady=(0, 10))

        # 2. Predicted yield card
        yield_frame = tk.Frame(results_grid, bg=BG_MAIN, bd=1, relief="solid", highlightthickness=0)
        yield_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        yield_lbl_title = tk.Label(yield_frame, text="ESTIMATED CROP YIELD", bg=BG_MAIN, fg=ACCENT_GREEN, font=("Helvetica", 9, "bold"))
        yield_lbl_title.pack(anchor="w", padx=12, pady=(10, 2))
        self.yield_res_val = tk.Label(yield_frame, text="Waiting for analysis...", bg=BG_MAIN, fg=TEXT_WHITE, font=("Helvetica", 14, "bold"))
        self.yield_res_val.pack(anchor="w", padx=12, pady=(0, 2))
        self.yield_bounds_val = tk.Label(yield_frame, text="", bg=BG_MAIN, fg=TEXT_SILVER, font=("Helvetica", 8, "italic"))
        self.yield_bounds_val.pack(anchor="w", padx=12, pady=(0, 10))

        # 3. K-Means soil cluster and guidance card
        soil_frame = tk.Frame(results_grid, bg=BG_MAIN, bd=1, relief="solid", highlightthickness=0)
        soil_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        soil_top_frame = tk.Frame(soil_frame, bg=BG_MAIN)
        soil_top_frame.pack(fill="x")
        soil_lbl_title = tk.Label(soil_top_frame, text="SOIL PROFILE SEGMENTATION ZONING", bg=BG_MAIN, fg=ACCENT_PURPLE, font=("Helvetica", 9, "bold"))
        soil_lbl_title.pack(side="left", padx=12, pady=(10, 2))
        
        self.soil_res_val = tk.Label(soil_top_frame, text="", bg=BG_MAIN, fg=TEXT_WHITE, font=("Helvetica", 11, "bold"))
        self.soil_res_val.pack(side="right", padx=12, pady=(10, 2))

        self.soil_guidance_text = tk.Text(
            soil_frame, height=3, bg=BG_MAIN, fg=TEXT_SILVER,
            font=("Helvetica", 9), wrap="word", bd=0, highlightthickness=0
        )
        self.soil_guidance_text.pack(fill="both", expand=True, padx=15, pady=(5, 10))
        self.soil_guidance_text.insert("1.0", "Agronomic guidance will be generated based on soil cluster profiling...")
        self.soil_guidance_text.config(state="disabled")

        # Visualizations Dashboard Card
        viz_card = ttk.Frame(right_container, style="Card.TFrame", padding=10)
        viz_card.grid(row=1, column=0, sticky="nsew")
        
        self.notebook = ttk.Notebook(viz_card)
        self.notebook.pack(fill="both", expand=True)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)

        self.notebook.add(self.tab1, text="Feature Importance")
        self.notebook.add(self.tab2, text="Soil Cluster PCA")
        self.notebook.add(self.tab3, text="Residual Analysis")

        # Responsive image canvases inside tabs
        self.canvas_imp = ResponsivePlotCanvas(self.tab1, PLOT_FEATURE_IMPORTANCE, "Feature Importance")
        self.canvas_imp.pack(fill="both", expand=True)

        self.canvas_pca = ResponsivePlotCanvas(self.tab2, PLOT_CLUSTER_SCATTER, "Soil Cluster PCA Scatter")
        self.canvas_pca.pack(fill="both", expand=True)

        self.canvas_res = ResponsivePlotCanvas(self.tab3, PLOT_RESIDUAL, "Residual Analysis")
        self.canvas_res.pack(fill="both", expand=True)

    def load_preset(self):
        """Load a valid preset for immediate testing (Rice crop parameters)."""
        presets = {
            "N": "90", "P": "42", "K": "43", "ph": "6.5",
            "temperature": "20.8", "humidity": "82.0", "rainfall": "202.9"
        }
        for key, val in presets.items():
            self.input_vars[key].set(val)

    def run_predictions(self):
        """Sequential inference pipeline triggered by user submission."""
        if not self.models_loaded:
            messagebox.showerror("Error", "ML Models are not loaded. Cannot run inference.")
            return

        # Parse & validate inputs
        inputs = []
        raw_vals = {}
        for key in ALL_FEATURES:
            try:
                val = float(self.input_vars[key].get().strip())
                inputs.append(val)
                raw_vals[key] = val
            except ValueError:
                messagebox.showerror("Input Validation Error",
                                     f"Please enter a valid numeric value for '{key}'.")
                return

        if not (0 <= raw_vals["ph"] <= 14):
            messagebox.showerror("Validation Error", "Soil pH must be between 0 and 14.")
            return

        feature_vector = np.array(inputs).reshape(1, -1)

        try:
            # 1. Decision Tree — Crop Recommendation
            recommended_crop = self.dt_clf.predict(feature_vector)[0]
            self.crop_res_val.config(text=str(recommended_crop).upper())

            # 2. K-Means — Soil Cluster Assignment
            soil_vector = np.array([raw_vals["N"], raw_vals["P"], raw_vals["K"], raw_vals["ph"]]).reshape(1, -1)
            soil_scaled = self.soil_scaler.transform(soil_vector)
            assigned_cluster = int(self.kmeans.predict(soil_scaled)[0])
            self.soil_res_val.config(text=f"ZONE ID: {assigned_cluster}")

            # Agronomic guidance
            guidance_list = self.agronomic_guidance.get(str(assigned_cluster), [
                "Ensure standard macro-nutrient levels.",
                "Conduct regular soil quality assays."
            ])
            formatted_guidance = "\n".join([f"• {g}" for g in guidance_list])

            self.soil_guidance_text.config(state="normal")
            self.soil_guidance_text.delete("1.0", tk.END)
            self.soil_guidance_text.insert("1.0", formatted_guidance)
            self.soil_guidance_text.config(state="disabled")

            # 3. Linear Regression — Yield Forecast
            predicted_yield = self.lr_model.predict(feature_vector)[0]
            predicted_yield = max(0.1, predicted_yield)

            lower_bound = max(0.0, predicted_yield - 1.96 * self.residual_std)
            upper_bound = predicted_yield + 1.96 * self.residual_std

            self.yield_res_val.config(text=f"{predicted_yield:.2f} metric tons per hectare")
            self.yield_bounds_val.config(
                text=f"95% Confidence Range: [{lower_bound:.2f} — {upper_bound:.2f}] tons/ha"
            )

            # Trigger a re-render of canvas plots in case sizes changed during loading
            self.canvas_imp.on_resize(tk.Event())
            self.canvas_pca.on_resize(tk.Event())
            self.canvas_res.on_resize(tk.Event())

            logger.info(f"Inference complete: crop={recommended_crop}, "
                        f"cluster={assigned_cluster}, yield={predicted_yield:.2f}")

        except Exception as e:
            logger.error(f"Inference pipeline error: {e}")
            messagebox.showerror("Pipeline Failure", f"An error occurred during prediction:\n{e}")


if __name__ == "__main__":
    app = SmartAgriGUI()
    app.mainloop()
