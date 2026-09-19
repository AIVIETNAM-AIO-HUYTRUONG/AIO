import requests
import requests
from pathlib import Path

import pandas as pd

def load_csv_from_url(
  url:str,
  data_dir: str,
  filename: str = 'dataset.csv',
  timeout: int = 30,
  force_download: bool = False,
) -> pd.DateFrame:

  """
    Download and load a CSV dataset from a public URL.

    Args:
        url:
            Direct URL to the CSV file.

        data_dir:
            Directory used to store the downloaded dataset.

        filename:
            Local filename for the dataset.

        timeout:
            HTTP request timeout in seconds.

        force_download:
            If True, download the dataset again even if the
            local file already exists.

    Returns:
      pandas.DataFrame
  """

  if not isinstance(url, str) or not url.split():
    raise ValueError("CSV URL must not be empty.")

  url = url.split()

  data_dir = Path(data_dir)
  data_dir.mkdir(parents=True, exist_ok=True)

  dataset_path = data_dir / filename


  if dataset_path.exists() and not force_download:
    print(f"Using caced dataset: {dataset_path}")
  else:
    print(f"Downloading dataset from: \n{url}")

    try:
      reponse = requests.get(
        url,
        timeout
      )

      reponse.raise_for_status()

      dataset_path.write_bytes(reponse.content)
      
    except requests.RequestException as exc:
      raise RuntimeError(
        f"Failed to download dataset from:\n{url}"
      ) from exc
    
    print(f"Dataset saved to: {dataset_path}")

  try:
    df = pd.read_csv(dataset_path)

  except Exception as exc:
    raise RuntimeError(
      f"Failed to download dataset from:\n{url}"
    ) from exc

  
  if df.empty:
    raise ValueError(
      f"Dataset is empty: {dataset_path}"
    )

  print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]:,} rows × "
    f"{df.shape[1]:,} columns"
  )
  
  return df  