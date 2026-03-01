import pandas as pd
import argparse
import sys

def analyze_report(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    required_columns = ['final_state', 'score', 'total_nodes_generated']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV is missing one of the required columns: {required_columns}")
        sys.exit(1)

    total_problems = len(df)
    
    successful_df = df[df['score'] == 1.0]

    zero_shot_success = successful_df[
        (successful_df['final_state'] == 'zero-shot') |
        ((successful_df['final_state'] == 'sample') & (successful_df['iterations'] == 1))
    ]
    count_zero_shot = len(zero_shot_success)

    evolve_success = successful_df[
        (successful_df['final_state'] == 'evolve') |
        ((successful_df['final_state'] == 'sample') & (successful_df['iterations'] > 1))
    ].copy()
    
    evolve_success.loc[evolve_success['total_nodes_generated'].isna(), 'total_nodes_generated'] = \
        evolve_success.loc[evolve_success['total_nodes_generated'].isna(), 'iterations']
    
    def count_nodes(dataframe, min_n, max_n):
        if min_n == 0:
            return len(dataframe[dataframe['total_nodes_generated'] <= max_n])
        else:
            return len(dataframe[
                (dataframe['total_nodes_generated'] > min_n) & 
                (dataframe['total_nodes_generated'] <= max_n)
            ])

    count_15 = count_nodes(evolve_success, 0, 15)
    count_15_50 = count_nodes(evolve_success, 15, 50)
    count_50_100 = count_nodes(evolve_success, 50, 100)
    count_100_200 = count_nodes(evolve_success, 100, 200)
    count_200_300 = count_nodes(evolve_success, 200, 300)
    count_300_400 = count_nodes(evolve_success, 300, 400)
    count_400_500 = count_nodes(evolve_success, 400, 500)
    count_500_1000 = count_nodes(evolve_success, 500, 1000)

    print("-" * 40)
    print(f"Analysis of: {file_path}")
    print(f"Total problems in file: {total_problems}")
    print("-" * 40)
    print(f"1. Successful Zero-shot:                         {count_zero_shot}")
    print(f"2. Successful Evolve (<= 15 nodes):              {count_15}")
    print(f"3. Successful Evolve (> 15 & <= 50 nodes):       {count_15_50}")
    print(f"4. Successful Evolve (> 50 & <= 100 nodes):      {count_50_100}")
    print(f"5. Successful Evolve (> 100 & <= 200 nodes):     {count_100_200}")
    print(f"6. Successful Evolve (> 200 & <= 300 nodes):     {count_200_300}")
    print(f"7. Successful Evolve (> 300 & <= 400 nodes):     {count_300_400}")
    print(f"8. Successful Evolve (> 400 & <= 500 nodes):     {count_400_500}")
    print(f"9. Successful Evolve (> 500 & <= 1000 nodes):    {count_500_1000}")
    print("-" * 40)
    print(f"Failed Problems:                               {total_problems - len(successful_df)}")
    
    df_for_avg = df.copy()
    
    df_for_avg.loc[
        (df_for_avg['final_state'] == 'zero-shot') |
        ((df_for_avg['final_state'] == 'sample') & (df_for_avg['iterations'] == 1)),
        'node_count'
    ] = 1
    
    df_for_avg.loc[df_for_avg['final_state'] == 'evolve', 'node_count'] = \
        df_for_avg.loc[df_for_avg['final_state'] == 'evolve', 'total_nodes_generated']
    
    df_for_avg.loc[
        (df_for_avg['final_state'] == 'sample') & (df_for_avg['iterations'] > 1),
        'node_count'
    ] = df_for_avg.loc[
        (df_for_avg['final_state'] == 'sample') & (df_for_avg['iterations'] > 1),
        'iterations'
    ]
    
    # Exclude failed problems and problems with > 300 nodes for average calculations
    df_for_avg_non_failed = df_for_avg[
        (df_for_avg['final_state'] != 'fail') & 
        (df_for_avg['node_count'] <= 300)
    ]
    
    if len(df_for_avg_non_failed) > 0:
        average_nodes = df_for_avg_non_failed['node_count'].mean()
        print(f"Average Node Count (excluding failed & >300):  {average_nodes:.2f}")
    else:
        print(f"Average Node Count (excluding failed & >300):  N/A (all failed)")
    
    if 'total_output_tokens' in df.columns:
        df_for_tokens = df_for_avg[
            (df_for_avg['final_state'] != 'fail') & 
            (df_for_avg['node_count'] <= 300)
        ]
        valid_output_tokens = df_for_tokens['total_output_tokens'].dropna()
        if len(valid_output_tokens) > 0:
            average_output_tokens = valid_output_tokens.mean()
            print(f"Average Output Token Count (excluding failed & >300): {average_output_tokens:.2f}")
        else:
            print(f"Average Output Token Count (excluding failed & >300): N/A (no data)")
    else:
        print(f"Average Output Token Count (excluding failed & >300): N/A (column not found)")
    
    print("-" * 40)
    
    count_15 = count_nodes(evolve_success, 0, 15)
    count_50 = count_nodes(evolve_success, 0, 50)
    count_100 = count_nodes(evolve_success, 0, 100)
    count_200 = count_nodes(evolve_success, 0, 200)
    count_300 = count_nodes(evolve_success, 0, 300)
    count_400 = count_nodes(evolve_success, 0, 400)
    count_500 = count_nodes(evolve_success, 0, 500)
    count_1000 = count_nodes(evolve_success, 0, 1000)
    
    print("Cumulative Successful Evolve Counts (including zero-shot):")
    print(f" - <= 15 nodes:    {count_zero_shot + count_15}")
    print(f" - <= 50 nodes:    {count_zero_shot + count_50}")
    print(f" - <= 100 nodes:   {count_zero_shot + count_100}")
    print(f" - <= 200 nodes:   {count_zero_shot + count_200}")
    print(f" - <= 300 nodes:   {count_zero_shot + count_300}")
    print(f" - <= 400 nodes:   {count_zero_shot + count_400}")
    print(f" - <= 500 nodes:   {count_zero_shot + count_500}")
    print(f" - <= 1000 nodes:  {count_zero_shot + count_1000}")
    
    print("-" * 40)
    print("Pass Rates (%):")
    print(f" - Zero-shot:      {count_zero_shot / total_problems * 100:.2f}%")
    print(f" - Evolve <= 15:   {(count_zero_shot + count_15) / total_problems * 100:.2f}%")
    print(f" - Evolve <= 50:   {(count_zero_shot + count_50) / total_problems * 100:.2f}%")
    print(f" - Evolve <= 100:  {(count_zero_shot + count_100) / total_problems * 100:.2f}%")
    print(f" - Evolve <= 200:  {(count_zero_shot + count_200) / total_problems * 100:.2f}%")
    print(f" - Evolve <= 300:  {(count_zero_shot + count_300) / total_problems * 100:.2f}%")
    print(f" - Evolve <= 400:  {(count_zero_shot + count_400) / total_problems * 100:.2f}%")
    print(f" - Evolve <= 500:  {(count_zero_shot + count_500) / total_problems * 100:.2f}%")
    print(f" - Evolve <=1000:  {(count_zero_shot + count_1000) / total_problems * 100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CSV report for success rates.")
    parser.add_argument("csv_file", help="Path to the CSV file to analyze")
    
    args = parser.parse_args()
    
    analyze_report(args.csv_file)