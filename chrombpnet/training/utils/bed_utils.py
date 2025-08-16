import pandas as pd

def read_bed_with_summit(file_path):
    """
    Read BED file and handle both 3-column and 10-column formats.
    For 3-column BED files, automatically calculates summit as peak center.
    
    Args:
        file_path: Path to BED file
        
    Returns:
        DataFrame with columns: chr, start, end, summit (and other columns if present)
    """
    # Try to read and determine number of columns
    df = pd.read_csv(file_path, sep='\t', header=None)
    
    if df.shape[1] == 3:
        # 3-column BED file: chr, start, end
        df.columns = ["chr", "start", "end"]
        # Calculate summit as peak center
        df["summit"] = (df["end"] - df["start"]) // 2
    elif df.shape[1] >= 10:
        # 10-column BED file with summit in the last column
        df.columns = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]
    else:
        # Handle other formats by adding missing columns and calculating summit
        base_columns = ["chr", "start", "end"]
        extra_columns = [str(i) for i in range(1, df.shape[1] - 2)]
        df.columns = base_columns + extra_columns
        # Calculate summit as peak center
        df["summit"] = (df["end"] - df["start"]) // 2
    
    return df