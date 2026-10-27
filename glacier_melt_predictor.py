"""
=============================================================
  GLACIER MELT RATE PREDICTOR
  Numerical Analysis Project — CP1
  Rhône Glacier, Switzerland (2015–2025)

  Methods:
    1. Edge Detection      — Canny + HSV pixel classification
    2. Nonlinear Curve Fitting — Least squares exponential decay
    3. Finite Differences  — Melt rate & acceleration
    4. Newton-Raphson      — Multi-threshold critical year prediction

  Run:
    pip install -r requirements.txt
    python glacier_melt_predictor.py

  All generated images are written to outputs/.
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import cv2

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

IMAGE_FILES = {
    2015:    "images/2015-08-29.jpg",
    2018:    "images/2018-08-28.jpg",
    2020:    "images/2020-08-27.jpg",
    2023:    "images/2023-08-22.jpg",
    2023.5:  "images/2023-08-24.jpg",
    2023.75: "images/2023-09-26.jpg",
    2025:    "images/2025-08-08.jpg",
}

THRESHOLDS = [0.70, 0.60, 0.50, 0.40]

OUTPUT_DIR = "outputs"

# ── Scale factor: pixels → km² ──
# Images are 1086x730 px and cover ~18km x 12km geographic area
IMAGE_KM2_TOTAL = 216.0
IMAGE_TOTAL_PX  = 1086 * 730
KM2_PER_PIXEL   = IMAGE_KM2_TOTAL / IMAGE_TOTAL_PX  # ~0.000273 km²/px

def px_to_km2(pixels):
    return pixels * KM2_PER_PIXEL

# ─────────────────────────────────────────────
#  STEP 1 — EDGE DETECTION + AREA EXTRACTION
# ─────────────────────────────────────────────

def extract_glacier_area(image_path, year, debug=False):
    """
    Extract glacier tongue area using HSV pixel classification
    combined with blue-grey channel analysis.

    Also generates a Canny edge visualization for the 2015
    baseline image to demonstrate edge detection methodology.

    Pipeline:
      1. Crop ROI to glacier tongue region
      2. HSV thresholding: medium brightness, low saturation
      3. Blue-grey filter: B channel >= R channel (ice vs rock)
      4. Morphological cleanup
      5. Pixel count = glacier area
      6. (2015 only) Canny edge detection visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    x1 = int(w * 0.37)
    x2 = int(w * 0.52)
    y1 = int(h * 0.40)
    y2 = int(h * 0.85)
    roi = img[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(roi)
    blue_grey_mask = (b.astype(int) >= r.astype(int) - 5).astype(np.uint8) * 255
    hsv_mask = cv2.inRange(hsv, np.array([0, 0, 110]), np.array([179, 35, 200]))
    mask = cv2.bitwise_and(hsv_mask, blue_grey_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5),  np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    area_pixels = int(np.sum(mask > 0))

    if int(year) == 2015:
        gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges    = cv2.Canny(blurred, threshold1=40, threshold2=100)

        canny_vis = roi.copy()
        canny_vis[edges > 0] = [0, 255, 255]
        canny_vis[mask  > 0] = [0, 200,   0]

        full_canny = img.copy()
        cv2.rectangle(full_canny, (x1, y1), (x2, y2), (255, 100, 0), 3)
        full_canny[y1:y2, x1:x2] = canny_vis
        canny_path = os.path.join(OUTPUT_DIR, "canny_2015_visualization.jpg")
        cv2.imwrite(canny_path, full_canny)
        print(f"  [Canny] Saved {canny_path}")

    if debug:
        full_debug = img.copy()
        cv2.rectangle(full_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
        colored = np.zeros_like(roi)
        colored[mask > 0] = [0, 255, 0]
        full_debug[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.6, colored, 0.4, 0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"debug_{int(year)}.jpg"), full_debug)

    return area_pixels


# ─────────────────────────────────────────────
#  STEP 2 — NONLINEAR CURVE FITTING
# ─────────────────────────────────────────────

def fit_melt_curve(years, areas):
    years = np.array(years, dtype=float)
    areas = np.array(areas, dtype=float)
    t = years - years[0]
    A = areas

    a0 = max(A[0] - A[-1], 500)
    b0 = 0.05
    c0 = max(min(A) * 0.5, 100)

    try:
        params, covariance = curve_fit(
            lambda t, a, b, c: a * np.exp(-b * t) + c,
            t, A,
            p0=[a0, b0, c0],
            maxfev=10000,
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
        )
    except RuntimeError:
        params     = [a0, b0, c0]
        covariance = np.zeros((3, 3))

    a, b, c = params

    A_pred    = a * np.exp(-b * t) + c
    ss_res    = np.sum((A - A_pred) ** 2)
    ss_tot    = np.sum((A - np.mean(A)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    t_smooth     = np.linspace(0, t[-1] + 25, 500)
    A_smooth     = a * np.exp(-b * t_smooth) + c
    years_smooth = t_smooth + years[0]

    perr    = np.sqrt(np.diag(covariance)) if covariance is not None else np.zeros(3)
    perr    = np.minimum(perr, np.array([a, b+1e-9, c+1e-9]) * 0.3)
    da, db, dc = perr
    A_upper = (a + da) * np.exp(-(b - db) * t_smooth) + (c + dc)
    A_lower = (a - da) * np.exp(-(b + db) * t_smooth) + (c - dc)
    A_lower = np.maximum(A_lower, 0)
    A_upper = np.minimum(A_upper, A[0] * 1.5)

    return params, years_smooth, A_smooth, A_upper, A_lower, years[0], r_squared


# ─────────────────────────────────────────────
#  STEP 3 — FINITE DIFFERENCES
# ─────────────────────────────────────────────

def finite_differences(years, areas):
    years   = np.array(years, dtype=float)
    areas   = np.array(areas, dtype=float)
    n       = len(years)
    dA_dt   = np.zeros(n)
    d2A_dt2 = np.zeros(n)

    for i in range(n):
        if i == 0:
            h        = years[1] - years[0]
            dA_dt[i] = (areas[1] - areas[0]) / h
        elif i == n - 1:
            h        = years[-1] - years[-2]
            dA_dt[i] = (areas[-1] - areas[-2]) / h
        else:
            # Non-uniform 3-point formula: h1 = backward step, h2 = forward step
            h1       = years[i]   - years[i-1]
            h2       = years[i+1] - years[i]
            dA_dt[i] = (h1**2 * areas[i+1] - h2**2 * areas[i-1]
                        + (h2**2 - h1**2) * areas[i]) / (h1 * h2 * (h1 + h2))

    for i in range(n):
        if i == 0 or i == n - 1:
            d2A_dt2[i] = np.nan
        else:
            # Non-uniform 3-point formula: h1 = backward step, h2 = forward step
            h1          = years[i]   - years[i-1]
            h2          = years[i+1] - years[i]
            d2A_dt2[i]  = 2 * ((areas[i+1] - areas[i]) / h2
                               - (areas[i] - areas[i-1]) / h1) / (h1 + h2)

    return dA_dt, d2A_dt2


# ─────────────────────────────────────────────
#  STEP 4 — NEWTON-RAPHSON (MULTIPLE THRESHOLDS)
# ─────────────────────────────────────────────

def newton_raphson(params, t0, A_critical, t_start, label="", tol=1e-6, max_iter=100):
    a, b, c = params

    def f(t):  return a * np.exp(-b * (t - t0)) + c - A_critical
    def fp(t): return -a * b * np.exp(-b * (t - t0))

    t = float(t_start)
    print(f"\n  ── Newton-Raphson: {label} ──")
    print(f"  A_critical = {A_critical:.0f} px²  ({px_to_km2(A_critical):.4f} km²)")
    fp_header = "f'(t)"
    print(f"  {'Iter':>4}  {'t':>10}  {'f(t)':>14}  {fp_header:>14}")
    print(f"  {'----':>4}  {'----------':>10}  {'----------':>14}  {'----------':>14}")

    for i in range(max_iter):
        ft  = f(t)
        fpt = fp(t)
        print(f"  {i:>4}  {t:>10.4f}  {ft:>14.4f}  {fpt:>14.6f}")
        if abs(fpt) < 1e-12:
            print("  Warning: derivative near zero.")
            break
        t_new = t - ft / fpt
        if abs(t_new - t) < tol:
            t = t_new
            print(f"  ✓ Converged in {i+1} iterations → {t:.2f}")
            break
        t = t_new
    return t


# ─────────────────────────────────────────────
#  STEP 5 — VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(years_data, areas_data, params, years_smooth, areas_smooth,
                 A_upper, A_lower, t0, dA_dt, d2A_dt2,
                 critical_years, thresholds, A_ref, r_squared):

    # Convert all areas to km² for plotting
    areas_data_km2   = [px_to_km2(a) for a in areas_data]
    areas_smooth_km2 = px_to_km2(areas_smooth)
    A_upper_km2      = px_to_km2(A_upper)
    A_lower_km2      = px_to_km2(A_lower)
    A_ref_km2        = px_to_km2(A_ref)
    dA_dt_km2        = px_to_km2(dA_dt)
    d2A_dt2_km2      = px_to_km2(d2A_dt2)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    C_ICE    = '#a8d8ea'
    C_RED    = '#ff6b6b'
    C_GREEN  = '#56cc9d'
    C_GOLD   = '#ffd166'
    C_PURPLE = '#c77dff'
    C_GRID   = '#1e2a35'
    C_TEXT   = '#e0e6ed'
    BG       = '#111820'
    T_COLORS = [C_GOLD, C_GREEN, C_RED, C_PURPLE]

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=C_TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_GRID)
        ax.grid(True, color=C_GRID, linewidth=0.7, alpha=0.8)

    # ── Plot 1: Area in km² ──
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)
    ax1.fill_between(years_smooth, A_lower_km2, A_upper_km2,
                     alpha=0.20, color=C_ICE, label='Uncertainty band (±1 std. error)')
    ax1.fill_between(years_smooth, areas_smooth_km2, alpha=0.12, color=C_ICE)
    ax1.plot(years_smooth, areas_smooth_km2, color=C_ICE, linewidth=2.5,
             label=f'Fitted Exponential Decay  (R² = {r_squared:.4f})')
    ax1.scatter(years_data, areas_data_km2, color=C_RED, s=120, zorder=5,
                label='Measured Area (edge detection)',
                edgecolors='white', linewidths=1.0)

    for i, (frac, crit_year) in enumerate(zip(thresholds, critical_years)):
        A_thresh_km2 = A_ref_km2 * frac
        if crit_year is not None and 2015 < crit_year < 2100:
            ax1.axhline(y=A_thresh_km2, color=T_COLORS[i],
                        linestyle='--', linewidth=1.5, alpha=0.8,
                        label=f'{int(frac*100)}% threshold → {crit_year:.1f}')
            ax1.axvline(x=crit_year, color=T_COLORS[i],
                        linestyle=':', linewidth=1.5, alpha=0.6)

    ax1.set_title('Rhône Glacier Area Over Time — Nonlinear Curve Fitting with Uncertainty Band',
                  color=C_TEXT, fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Year', color=C_TEXT, fontsize=11)
    ax1.set_ylabel('Glacier Area (km²)', color=C_TEXT, fontsize=11)
    ax1.legend(facecolor='#1a2530', edgecolor=C_GRID,
               labelcolor=C_TEXT, fontsize=9, loc='upper right')

    # ── Plot 2: Melt Rate in km²/year ──
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    ax2.bar(years_data, np.abs(dA_dt_km2), color=C_RED, alpha=0.75,
            edgecolor=C_RED, linewidth=0.5)
    ax2.plot(years_data, np.abs(dA_dt_km2), color='white',
             linewidth=1.5, marker='o', markersize=5)
    ax2.set_title('Melt Rate — 1st Derivative (Finite Differences)',
                  color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Year', color=C_TEXT) 
    ax2.set_ylabel('|dA/dt| (km²/year)', color=C_TEXT)

    # ── Plot 3: Acceleration in km²/year² ──
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    valid  = ~np.isnan(d2A_dt2_km2)
    c_bars = [C_RED if v < 0 else C_GREEN for v in d2A_dt2_km2[valid]]
    ax3.bar(np.array(years_data)[valid], d2A_dt2_km2[valid],
            color=c_bars, alpha=0.75)
    ax3.axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.5)
    ax3.set_title('Melt Acceleration — 2nd Derivative (Finite Differences)',
                  color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
    ax3.set_xlabel('Year', color=C_TEXT)
    ax3.set_ylabel('d²A/dt² (km²/year²)', color=C_TEXT)

    # ── Plot 4: Newton-Raphson f(t) ──
    ax4 = fig.add_subplot(gs[2, :])
    style_ax(ax4)
    t_range = np.linspace(2020, 2060, 500)
    a, b, c = params
    f_vals = px_to_km2(a * np.exp(-b * (t_range - t0)) + c - A_ref * 0.50)
    ax4.plot(t_range, f_vals, color=C_ICE, linewidth=2,
             label='f(t) = A(t) − A_critical (50% threshold)')
    ax4.axhline(y=0, color='white', linewidth=1.5,
                linestyle='--', alpha=0.7, label='f(t) = 0 (root)')
    ax4.fill_between(t_range, f_vals, 0,
                     where=(f_vals >= 0), alpha=0.15, color=C_GREEN)
    ax4.fill_between(t_range, f_vals, 0,
                     where=(f_vals < 0),  alpha=0.15, color=C_RED)

    for i, (frac, crit_year) in enumerate(zip(thresholds, critical_years)):
        if crit_year is not None and 2015 < crit_year < 2100:
            ax4.axvline(x=crit_year, color=T_COLORS[i],
                        linewidth=2, linestyle=':',
                        label=f'Root {int(frac*100)}%: t = {crit_year:.2f}')

    ax4.set_title('Newton-Raphson Root Finding — f(t) = A(t) − A_critical',
                  color=C_TEXT, fontsize=11, fontweight='bold', pad=10)
    ax4.set_xlabel('Year', color=C_TEXT)
    ax4.set_ylabel('f(t) (km²)', color=C_TEXT)
    ax4.legend(facecolor='#1a2530', edgecolor=C_GRID,
               labelcolor=C_TEXT, fontsize=9)

    fig.suptitle('GLACIER MELT RATE PREDICTOR — Rhône Glacier, Switzerland',
                 color='white', fontsize=16, fontweight='bold', y=1.01)

    plot_path = os.path.join(OUTPUT_DIR, 'glacier_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {plot_path}")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  GLACIER MELT RATE PREDICTOR — Rhône Glacier, Switzerland")
    print("  Numerical Analysis CP1")
    print("=" * 62)
    print(f"\n  Scale factor: 1 pixel = {KM2_PER_PIXEL:.6f} km²")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[STEP 1] Edge Detection → Glacier Area Extraction\n")
    years_raw, areas_raw = [], []

    for year, path in sorted(IMAGE_FILES.items()):
        if not os.path.exists(path):
            print(f"  SKIP: {path}")
            continue
        try:
            area = extract_glacier_area(path, year, debug=True)
            print(f"  {year:.2f}  →  {area:,.0f} px²  ({px_to_km2(area):.3f} km²)")
            years_raw.append(year)
            areas_raw.append(area)
        except Exception as e:
            print(f"  ERROR {path}: {e}")

    if len(years_raw) < 4:
        print("Need at least 4 images.")
        return

    years_raw = np.array(years_raw)
    areas_raw = np.array(areas_raw)

    print("\n  Averaging same-year images...")
    year_int     = np.floor(years_raw).astype(int)
    unique_years = np.unique(year_int)
    years_avg, areas_avg = [], []

    for y in unique_years:
        mask = year_int == y
        years_avg.append(float(y))
        areas_avg.append(np.mean(areas_raw[mask]))
        print(f"  {y}: {areas_avg[-1]:,.0f} px²  =  {px_to_km2(areas_avg[-1]):.3f} km²  ({mask.sum()} image(s))")

    years_avg = np.array(years_avg)
    areas_avg = np.array(areas_avg)

    print("\n[STEP 2] Nonlinear Curve Fitting — Exponential Decay\n")
    params, years_smooth, areas_smooth, A_upper, A_lower, t0, r_sq = \
        fit_melt_curve(years_avg, areas_avg)
    a, b, c = params
    print(f"  A(t) = {a:.2f} · exp(−{b:.5f} · (t − {t0:.0f})) + {c:.2f}  [px²]")
    print(f"  A(t) = {px_to_km2(a):.4f} · exp(−{b:.5f} · (t − {t0:.0f})) + {px_to_km2(c):.4f}  [km²]")
    print(f"  Decay rate:  {b:.5f} /yr → {(1-np.exp(-b))*100:.2f}% per year")
    print(f"  R² score:    {r_sq:.4f}")

    print("\n[STEP 3] Finite Differences — Derivatives\n")
    dA_dt, d2A_dt2 = finite_differences(years_avg, areas_avg)
    print(f"  {'Year':>6}  {'dA/dt (km²/yr)':>16}  {'d²A/dt² (km²/yr²)':>20}")
    print(f"  {'------':>6}  {'---------------':>16}  {'------------------':>20}")
    for i in range(len(years_avg)):
        acc = f"{px_to_km2(d2A_dt2[i]):.5f}" if not np.isnan(d2A_dt2[i]) else "N/A"
        print(f"  {years_avg[i]:>6.0f}  {px_to_km2(dA_dt[i]):>16.5f}  {acc:>20}")

    valid_idx = np.where(~np.isnan(d2A_dt2))[0]
    if len(valid_idx):
        mx = valid_idx[np.argmax(np.abs(d2A_dt2[valid_idx]))]
        print(f"\n  ⚡ Max melt acceleration at: {years_avg[mx]:.0f}")
    print(f"\n  Note: Endpoint derivatives use one-sided differences")
    print(f"        and are less accurate than central differences.")
    print(f"        2nd derivatives interpreted qualitatively due to noise.")

    print("\n[STEP 4] Newton-Raphson — Multi-Threshold Root Finding\n")
    A_ref = areas_avg[0]
    print(f"  Reference area (2015): {A_ref:,.0f} px²  =  {px_to_km2(A_ref):.3f} km²")

    critical_years = []
    for frac in THRESHOLDS:
        A_crit = A_ref * frac
        if c >= A_crit:
            # The fitted curve never drops below its asymptote c, so this
            # threshold has no root — report it instead of moving the target.
            print(f"\n  ── {int(frac*100)}% of 2015 area ──")
            print(f"  A_critical = {A_crit:.0f} px² ≤ asymptote c = {c:.0f} px²"
                  f" → not reached under this model")
            critical_years.append(None)
            continue
        crit = newton_raphson(
            params, t0, A_crit,
            t_start=years_avg[-1] + 5,
            label=f"{int(frac*100)}% of 2015 area"
        )
        critical_years.append(crit)

    print("\n[STEP 5] Generating plots...\n")
    plot_results(years_avg, areas_avg, params,
                 years_smooth, areas_smooth, A_upper, A_lower,
                 t0, dA_dt, d2A_dt2,
                 critical_years, THRESHOLDS, A_ref, r_sq)

    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    print(f"  Decay rate:         {b:.5f} /yr  ({(1-np.exp(-b))*100:.2f}% per year)")
    print(f"  R² fit quality:     {r_sq:.4f}")
    print(f"  2015 area:          {areas_avg[0]:,.0f} px²  =  {px_to_km2(areas_avg[0]):.3f} km²")
    print(f"  2025 area:          {areas_avg[-1]:,.0f} px²  =  {px_to_km2(areas_avg[-1]):.3f} km²")
    pct = (1 - areas_avg[-1] / areas_avg[0]) * 100
    print(f"  Area lost 2015→2025: {pct:.1f}%")
    print()
    for frac, cy in zip(THRESHOLDS, critical_years):
        A_thresh = A_ref * frac
        result = f"{cy:.2f}" if cy is not None else "not reached under this model"
        print(f"  {int(frac*100)}% threshold ({px_to_km2(A_thresh):.3f} km²) → critical year: {result}")
    print("=" * 62)


if __name__ == "__main__":
    main()
