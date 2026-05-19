# import pandas as pd
# import datetime
# import random
# from pathlib import Path
# import re

# from .download import download_csic_2010

# def setup_project_data(data_dir="./data/csic-2010", force_rebuild=False):
#     """
#     Orchestrates the download, cleaning, and augmentation of the CSIC dataset.
#     Ensures that the pipeline only runs if the final files are missing.
#     """
#     base_path = Path(data_dir)
#     raw_csv = base_path / "csic_database.csv"
#     train_out = Path(base_path / "train_dataset.csv")
#     test_out = Path(base_path / "test_dataset.csv")

#     # 1. Check if final datasets already exist
#     if train_out.exists() and test_out.exists() and not force_rebuild:
#         print("Train and Test datasets already exist")
#         return train_out, test_out

#     # 2. Check if raw data exists, if not, download it
#     if not raw_csv.exists():
#         print(f"CSIC data not found at {raw_csv}. Starting download.")
#         base_path.mkdir(parents=True, exist_ok=True)
#         download_csic_2010(base_path) 
#     else:
#         print("Raw CSV found")

#     # 2.5: Change content length column
#     formatted_df = _format_data(raw_csv)

#     # 3. Generate the augmented datasets
#     print("Generating ZScaler style data")
#     try:
#         generate_augmented_files(formatted_df, train_out, test_out)
#         print(f"Created {train_out} and {test_out}")
#     except Exception as e:
#         print(f"Error during generation: {e}")

#     return train_out, test_out


# def generate_augmented_files(df, train_path, test_path):
#     normal_df = df[df['classification'] == 0]
#     anomaly_df = df[df['classification'] == 1]
    
#     # Do the splits (don't shuffle them together yet, we want to weave them)
#     train_normal = normal_df.sample(frac=0.8, random_state=42)
#     train_anomaly = anomaly_df.sample(frac=0.8, random_state=42)
    
#     test_normal = normal_df.drop(train_normal.index)
#     test_anomaly = anomaly_df.drop(train_anomaly.index)

#     base_time = datetime.datetime(2026, 5, 8, 9, 0, 0)
    
#     train_final = apply_nss_metadata(train_normal, train_anomaly, base_time)
#     test_final = apply_nss_metadata(test_normal, test_anomaly, base_time + datetime.timedelta(days=7))

#     # Save WITHOUT shuffling
#     train_final.to_csv(train_path, index=False)
#     test_final.to_csv(test_path, index=False)

# def apply_nss_metadata(normal_df, anomaly_df, start_time):
#     users = [f"user_{i:02d}@enterprise.com" for i in range(1, 51)]
#     normal_ips = [f"10.10.1.{i}" for i in range(10, 60)]
#     attacker_ips = ["192.168.5.99", "192.168.5.100"]
    
#     current_time = start_time
#     rows = []
    
#     # Convert to lists for easier popping
#     normal_records = normal_df.to_dict('records')
#     anomaly_records = anomaly_df.to_dict('records')
    
#     while normal_records or anomaly_records:
#         # 5% chance an attack bursts in, otherwise normal traffic
#         if anomaly_records and (not normal_records or random.random() < 0.05):
#             # Attack Burst: 1 to 5 requests in very rapid succession
#             burst_size = random.randint(1, 5)
            
#             # Attackers usually stick to one IP/User per burst
#             # Sometimes they use a known bad IP, sometimes a compromised normal IP
#             burst_ip = random.choice(attacker_ips) if random.random() < 0.7 else random.choice(normal_ips)
#             burst_user = random.choice(users)
            
#             for _ in range(burst_size):
#                 if not anomaly_records: break
#                 row = anomaly_records.pop(0)
                
#                 # Attacks happen fast (0 to 1 seconds apart)
#                 current_time += datetime.timedelta(milliseconds=random.randint(50, 1000))
                
#                 rows.append({
#                     'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], # include milliseconds
#                     'user': burst_user,
#                     'client_ip': burst_ip,
#                     'method': row['Method'],
#                     'url': str(row['URL']).split(' ')[0],
#                     'content_length': row['lenght'],
#                     'label': row['classification']
#                 })
#         else:
#             # Normal Traffic
#             row = normal_records.pop(0)
            
#             # Normal traffic trickles in slower (1 to 15 seconds apart)
#             current_time += datetime.timedelta(seconds=random.randint(1, 15))
            
#             rows.append({
#                 'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
#                 'user': random.choice(users),
#                 'client_ip': random.choice(normal_ips),
#                 'method': row['Method'],
#                 'url': str(row['URL']).split(' ')[0],
#                 'content_length': row['lenght'],
#                 'label': row['classification']
#             })
            
#     return pd.DataFrame(rows)

# if __name__ == "__main__":
#     train_out, test_out = setup_project_data()