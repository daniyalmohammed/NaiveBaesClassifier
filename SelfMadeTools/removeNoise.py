import pandas as pd

# Load the original CSV file into a pandas dataframe
df = pd.read_csv('new_train.csv')

start_patterns_soci = ["SOCIAL HISTORY:,", "SOCIAL HISTORY: ,", "SOCIAL HISTORY:  ,", "SOCIAL HISTORY:   ,", "SOCIAL HISTORY,", "SOCIAL HISTORY ,", "SOCIAL HISTORY:"]
start_patterns_fam = ["FAMILY HISTORY:,", "FAMILY HISTORY: ,", "FAMILY HISTORY:  ,", "FAMILY HISTORY:   ,", "FAMILY HISTORY,", "FAMILY HISTORY ,", "FAMILY HISTORY:"]
start_patterns_med = ["MEDICAL HISTORY:,", "MEDICAL HISTORY: ,", "MEDICAL HISTORY:  ,", "MEDICAL HISTORY:   ,", "MEDICAL HISTORY,", "MEDICAL HISTORY ,", "MEDICAL HISTORY:"]
start_patterns_diag = ["PREOPERATIVE DIAGNOSIS:,", "PREOPERATIVE DIAGNOSIS:, ,", "PREOPERATIVE DIAGNOSIS:,  ,", "PREOPERATIVE DIAGNOSIS:,   ,", "PREOPERATIVE DIAGNOSIS:,", "PREOPERATIVE DIAGNOSIS: ,", "PREOPERATIVE DIAGNOSIS:,"]
start_patterns = [start_patterns_soci, start_patterns_fam, start_patterns_med, start_patterns_diag]

def remove_hist(transcription):
    for start_patterns_list in start_patterns:
        # Find the start position of the block of text
        start = -1
        dummy = 0
        for pattern in start_patterns_list:
            start = transcription.find(pattern)
            if start != -1:
                dummy = len(pattern)
                break
    
        # Check if the start position is valid
        if start != -1:
            # Find the end position of the block of text
            end = transcription.find(",", start+dummy)
            if end == -1:
                end = len(transcription)
        

            # Extract the parts of the string that come before and after the block of text
            before = transcription[:start]
            after = transcription[end+1:]
            
            # Join the remaining parts of the string
            new_string = before + after
            transcription = new_string + "pool"
        else:
            continue
    return transcription

# Apply the function to the specific column, creating a new column with the modified text
df['transcription'] = df['transcription'].apply(remove_hist)

# Save the modified dataframe as a new CSV file
df.to_csv('modified2.csv', index=False)
