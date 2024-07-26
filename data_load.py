import pandas as pd

def load_dataset(name):
    if name == 'data_h':
        return get_h()
    elif name == 'data_d':
        return get_d()
    else:
        return get_XXX()

def get_h():
    # Load the dataset
    file_path = "data_preprocessing/processed_data/data_h.csv"
    df = pd.read_csv(file_path)
    times = df['date']
    # Convert pandas dataframe to numpy array with float64
    series = df['val_total_power'].astype('float64').values 
    
    return series,times

def get_d():
    # Load the dataset
    file_path = "data_preprocessing/processed_data/data_d.csv"
    df = pd.read_csv(file_path)
    times = df['date']
    # Convert pandas dataframe to numpy array with float64
    series = df['val_total_power'].astype('float64').values 
    
    return series,times

def get_XXX():
    # not implemented
    pass