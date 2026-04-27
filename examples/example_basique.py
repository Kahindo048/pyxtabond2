"""
Usage example of the PyXtabond2 package.

This script demonstrates how to estimate dynamic panel models (GMM),
from the basic specification (Arellano-Bond) to advanced models 
including the Windmeijer correction, Forward Orthogonal Deviations (FOD), 
and instrument compression (collapse).

It also shows how to compare these models and export the results
for academic publication using GMMStargazer.
"""

import pandas as pd
from pyxtabond2.data_utils import PanelData
from pyxtabond2.api import PyXtabond2
from pyxtabond2.exporter import GMMStargazer
from pyxtabond2.load_data import load_dataset

def main():
    print("--- Starting PyXtabond2 demonstration ---\n")

    # 1. Loading the data
    # Make sure the df_panel.xlsx file is in the same folder
    try:
        df = load_dataset('df_panel.xlsx')
        print(f"Data loaded successfully: {df.shape[0]} observations.")
    except FileNotFoundError:
        print("Error: The file 'df_panel.xlsx' could not be found.")
        return

    # Data preparation
    panel = PanelData(df, id_col='Country', time_col='Year')
    panel.data['L1_Growth'] = panel.get_lag('Growth', 1)
    df_ready = panel.data.reset_index()

    id_col = 'Country'          # Group identifier (country, firm)
    time_col = 'Year'           # Time identifier
    dep_var = 'Growth'          # Dependent variable
    x_vars = ['Capital', 'Labor', 'Wage', 'Investment', 'Ide'] # Explanatory variables
    gmm_vars = ['Growth', 'Capital'] # Variables for Arellano-Bond instruments
    iv_vars = ['Ide']           # Variables for standard instruments

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
                        model_type='difference',
                        twostep=False)
    
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
    # MODEL 5: System GMM with Forward Orthogonal Deviations (FOD)
    # Ideal for panels with gaps (missing years)
    # ==========================================
    print("Estimating Model 5: System GMM (FOD + Collapsed)...")
    model5 = PyXtabond2(df_ready, 
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
                        collapse=True,
                        orthogonal=True) # Replaces first difference with orthogonal deviation
    
    res5 = model5.fit()
    res5.summary()

    # ==========================================
    # MODEL 6: Difference GMM With IFE (Arellano-Bond)
    # ==========================================
    print("\nEstimating Model 6: Difference GMM With IFE (Arellano-Bond)")
    model6 = PyXtabond2(df_ready, 
                        id_col = 'Country', 
                        time_col = 'Year', 
                        dep_var = 'Growth', 
                        x_vars = ['L1_Growth', 'Capital', 'Labor', 'Wage', 'Investment', 'Ide'], 
                        gmm_vars =['Growth', 'Capital'], 
                        iv_vars = ['Ide'],
                        model_type='difference',
                        twostep=False,
                        r = 2)
    
    res6 = model6.fit()
    res6.summary()

    # ==========================================
    # COMPARISON AND EXPORT OF RESULTS
    # ==========================================
    print("\n--- Generating comparative table ---")
    
    # We gather our models in the Stargazer class
    models_to_compare = [res1, res2, res3, res4, res5, res6]
    model_names = [
        "Diff (1-Step)", 
        "Sys (1-Step)", 
        "Sys (2-Step Rob)", 
        "Sys (Collapsed)", 
        "Sys (FOD)",
        "Diff IFE (1-Step)"
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