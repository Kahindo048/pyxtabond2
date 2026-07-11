"""
IFE-GMM worked example: automatic factor-count selection.

Companion to examples.py (the six-specification standard-GMM
workflow). This script exercises the interactive-fixed-effects extension on
the bundled df_ife_panel.csv dataset (built with a genuine multifactor error
structure, see the User Manual's "Bundled datasets" section) across four
specifications: fixed r=2 (one-step, then two-step Windmeijer-robust),
automatic factor-count selection (r='auto', inspecting the ER/ICp2 criteria
via plot_factor_selection), and the split-panel-jackknife bias-corrected fit.

Usage
-----
    python examples/ife_exemples.py
"""

import contextlib
import io

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, so this runs without a display

from pyxtabond2 import PanelData, PyXtabond2, GMMStargazer, load_dataset


def main():
    log = io.StringIO()

    def emit(msg=""):
        print(msg)
        log.write(str(msg) + "\n")
        
    print("--- Starting PyXtabond2 demonstration IFE-GMM ---\n")
    
    # 1. Loading the data
    try:
        df = load_dataset('df_ife_panel.csv')
        print(f"Data loaded successfully: {df.shape[0]} observations.")
    except FileNotFoundError:
        print("Error: The file 'df_ife_panel.csv' could not be found.")
        return

    # Data preparation
    panel = PanelData(df, id_col='Country', time_col='Year')
    panel.data['L1_Growth'] = panel.get_lag('Growth', 1)
    df_ready = panel.data.reset_index()

    # ==========================================
    # MODEL 1: System GMM With IFE (Blundell-Bond)
    # ==========================================
    print("Estimating Model 1: System GMM With IFE (1-Step)...")
    model1 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital'], 
                        gmm_vars =['Growth', 'Capital'], 
                        model_type='system',
                        r = 2)
    
    res1 = model1.fit()
    res1.summary()
    
    # ==========================================
    # MODEL 2: System GMM With IFE (Two-Step Robust)
    # Includes the Windmeijer correction for small samples
    # ==========================================
    print("Estimating Model 2: System GMM With IFE (2-Step Robust)...")
    model2 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital'], 
                        gmm_vars =['Growth', 'Capital'], 
                        model_type='system',
                        twostep=True,
                        robust=True,
                        small=True,
                        r = 2,) # Activates degrees of freedom corrections (t and F tests)
    
    res2 = model2.fit()
    res2.summary()
    
    # ==========================================
    # MODEL 3: System GMM with IFE (Automatic factor-count selection)
    # r is selected automatically via the Ahn-Horenstein (2013) ER and
    # Bai-Ng (2002) ICp2 criteria instead of being fixed by the user.
    # ==========================================
    print("Estimating Model 3: System GMM With IFE (r='auto')...")
    model3 = PyXtabond2(df_ready,
                        id_col = 'Country',
                        time_col = 'Year',
                        dep_var = 'Growth',
                        x_vars = ['L1_Growth', 'Capital'],
                        gmm_vars =['Growth', 'Capital'],
                        model_type='system',
                        twostep=True,
                        robust=True,
                        small=True,
                        r = 'auto',
                        r_max = 5) # Automatic factor-count selection (ER / ICp2 criteria)

    res3 = model3.fit()
    res3.summary()
    model3.plot_factor_selection(mode='table')  # ER/ICp2 criteria across candidate factor counts
    
    # ==========================================
    # MODEL 4: System GMM with IFE and "Collapsed Instruments"
    # Reduces the number of instruments to avoid over-identifying the model
    # ==========================================
    print("Estimating Model 4: System GMM With IFE (Collapsed)...")
    model4 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital'], 
                        gmm_vars =['Growth', 'Capital'], 
                        model_type='system',
                        twostep=True,
                        robust=True,
                        small=True,
                        collapse=True,
                        r = 2,
                        bias_correction=True) # Collapses the instrument matrix
    
    res4 = model4.fit()
    res4.summary()
    
    # ==========================================
    # COMPARISON AND EXPORT OF RESULTS
    # ==========================================
    print("\n--- Generating comparative table ---")
    
    # We gather our models in the Stargazer class
    models_to_compare = [res1, res2, res3, res4]
    model_names = [
        "1-Step",
        "2-Step Robust",
        "Auto r",
        "Bias Corrected"
    ]
    
    stargazer = GMMStargazer(models_to_compare, model_names=model_names)
    
    # Export to Word (Ideal for drafts and reviews)
    word_file = "gmm_ife_comparison.docx"
    stargazer.to_word(filepath=word_file)
    
    # Export to LaTeX (Ideal for final publication)
    tex_file = "gmm_ife_comparison.tex"
    latex_code = stargazer.to_latex(filepath=tex_file)
    
    print(f"\nResults were successfully exported to '{word_file}' and '{tex_file}'.")

if __name__ == "__main__":
    main()