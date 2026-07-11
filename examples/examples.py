"""
Usage example of the PyXtabond2 package.

This script demonstrates how to estimate dynamic panel models (GMM),
from the basic specification (Arellano-Bond) to advanced models
including the Windmeijer correction and instrument compression (collapse).

It also shows how to compare these models and export the results
for academic publication using GMMStargazer.
"""

import pandas as pd
from pyxtabond2 import PanelData, PyXtabond2, GMMStargazer, load_dataset

def main():
    print("--- Starting PyXtabond2 demonstration ---\n")

    # 1. Loading the data
    try:
        df = load_dataset('df_panel.csv')
        print(f"Data loaded successfully: {df.shape[0]} observations.")
    except FileNotFoundError:
        print("Error: The file 'df_panel.csv' could not be found.")
        return

    # Data preparation
    panel = PanelData(df, id_col='Country', time_col='Year')
    panel.data['L1_Growth'] = panel.get_lag('Growth', 1)
    df_ready = panel.data.reset_index()

    # ==========================================
    # MODEL 1: Difference GMM (Arellano-Bond)
    # ==========================================
    print("\nEstimating Model 1: Difference GMM (1-Step)...")
    model1 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital', 'Labor', 'Wage', 'Investment', 'Ide'], 
                        gmm_vars =['Growth', 'Capital'], 
                        iv_vars = ['Ide'],
                        model_type='difference')
    
    res1 = model1.fit()
    res1.summary()

    # ==========================================
    # MODEL 2: System GMM (Blundell-Bond)
    # ==========================================
    print("Estimating Model 2: System GMM (1-Step)...")
    model2 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital', 'Labor', 'Wage', 'Investment', 'Ide'], 
                        gmm_vars =['Growth', 'Capital'], 
                        iv_vars = ['Ide'],
                        model_type='system',
                        twostep=False)
    
    res2 = model2.fit()
    res2.summary()

    # ==========================================
    # MODEL 3: System GMM (Two-Step Robust)
    # Includes the Windmeijer correction for small samples
    # ==========================================
    print("Estimating Model 3: System GMM (2-Step Robust)...")
    model3 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital', 'Labor', 'Wage', 'Investment', 'Ide'], 
                        gmm_vars =['Growth', 'Capital'], 
                        iv_vars = ['Ide'],
                        model_type='system',
                        twostep=True,
                        robust=True,
                        small=True) # Activates degrees of freedom corrections (t and F tests)
    
    res3 = model3.fit()
    res3.summary()

    # ==========================================
    # MODEL 4: System GMM with "Collapsed Instruments"
    # Reduces the number of instruments to avoid over-identifying the model
    # ==========================================
    print("Estimating Model 4: System GMM (Collapsed)...")
    model4 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital', 'Labor', 'Wage', 'Investment', 'Ide'], 
                        gmm_vars =['Growth', 'Capital'], 
                        iv_vars = ['Ide'],
                        model_type='system',
                        twostep=True,
                        robust=True,
                        small=True,
                        collapse=True) # Collapses the instrument matrix
    
    res4 = model4.fit()
    res4.summary()

    

    # ==========================================
    # COMPARISON AND EXPORT OF RESULTS
    # ==========================================
    print("\n--- Generating comparative table ---")
    
    # We gather our models in the Stargazer class
    models_to_compare = [res1, res2, res3, res4]
    model_names = [
        "Diff (1-Step)", 
        "Sys (1-Step)", 
        "Sys (2-Step Rob)", 
        "Sys (Collapsed)"
    ]
    
    stargazer = GMMStargazer(models_to_compare, model_names=model_names)
    
    # Export to Word (Ideal for drafts and reviews)
    word_file = "gmm_comparison.docx"
    stargazer.to_word(filepath=word_file)
    
    # Export to LaTeX (Ideal for final publication)
    tex_file = "gmm_comparison.tex"
    latex_code = stargazer.to_latex(filepath=tex_file)
    
    print(f"\nResults were successfully exported to '{word_file}' and '{tex_file}'.")

if __name__ == "__main__":
    main()