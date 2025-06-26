import json
import pandas as pd

def fix_failed_output(df: pd.DataFrame, failed_output_col: str = 'failed_output') -> pd.DataFrame:
    if failed_output_col not in df:
        return df

    mask = df[failed_output_col].notnull()
    fixed_output = []
    for generated_text in df.loc[mask, failed_output_col]:
        if generated_text.endswith('"'):
            # reached max token 
            generated_text += '}'
        
        if not generated_text.endswith('"}'):
            # reached max token 
            generated_text += '"}'

        if generated_text.startswith('{'):
            # replace double quotes in reasoning with single quote
            start_idx = generated_text.find('"Reason": "') 
            if start_idx != -1:
                start_idx += len('"Reason": "')
                end_idx = -3
                generated_text = (
                    generated_text[:start_idx] 
                    + generated_text[start_idx:end_idx].replace('"', '\'') 
                    + generated_text[end_idx:]
                )

        try:
            result = json.loads(generated_text)
        except json.JSONDecodeError:
            result = {failed_output_col: generated_text}
        fixed_output.append(result)
    
    fixed_output = pd.DataFrame(fixed_output, index=df.index[mask])
    df.loc[mask, fixed_output.columns] = fixed_output
    return df