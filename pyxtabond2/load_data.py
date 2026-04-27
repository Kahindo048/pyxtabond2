import os
import pandas as pd

def list_datasets() -> list:
    """
    Displays and returns the list of example datasets included in PyXtabond2.
    
    Returns
    -------
    list
        A list containing the names of available data files.
    """
    base_dir = os.path.dirname(__file__)
    datasets_dir = os.path.join(base_dir, 'datasets')
    
    if not os.path.exists(datasets_dir):
        print("No datasets are currently available.")
        return []
        
    # List files ignoring hidden files (starting with '.') 
    # and Python special files (starting with '__')
    files = [f for f in os.listdir(datasets_dir) 
             if os.path.isfile(os.path.join(datasets_dir, f)) and not f.startswith(('.', '_'))]
    
    print("\n=== PyXtabond2 Example Datasets ===")
    if not files:
        print(" (The directory is empty)")
    else:
        for f in sorted(files):
            # You could later add a dictionary here to map 
            # file names to descriptions (e.g., "macro_panel.csv - Macroeconomic data")
            print(f" - {f}")
    print("===================================\n")
            
    return sorted(files)

def load_dataset(name: str) -> pd.DataFrame:
    """
    Loads an example dataset included in PyXtabond2.
    
    Parameters
    ----------
    name : str
        The name of the data file (e.g., 'macro_panel.csv' or 'firm_investments.dta').
        Use `pyxtabond2.list_datasets()` to view available options.
        
    Returns
    -------
    pd.DataFrame
        The loaded dataset ready for analysis.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist in the datasets directory.
    ValueError
        If the file format is not supported.
    """
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'datasets', name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The dataset '{name}' could not be found. "
            f"Use list_datasets() to view available files."
        )
        
    # Automatic format detection
    if name.endswith('.csv'):
        return pd.read_csv(file_path)
    elif name.endswith('.dta'):
        return pd.read_stata(file_path)
    elif name.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv, .dta, or .xlsx.")