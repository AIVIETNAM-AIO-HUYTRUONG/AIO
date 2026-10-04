from pathlib import Path

import numpy as np
import pandas as pd
import requests


def load_csv_from_url(
    url: str,
    data_dir: str | Path,
    filename: str = "dataset.csv",
    timeout: int = 30,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download and load a CSV dataset from a public URL.

    The dataset is cached locally. If the dataset already exists
    and force_download=False, the existing file will be reused.

    Args:
        url:
            Direct URL to the CSV file.

        data_dir:
            Directory where the dataset will be stored.

        filename:
            Local filename for the dataset.

        timeout:
            HTTP request timeout in seconds.

        force_download:
            If True, download the dataset again even if the
            local file already exists.

    Returns:
        pandas.DataFrame:
            Loaded dataset.

    Raises:
        ValueError:
            If the URL is invalid or the dataset is empty.

        RuntimeError:
            If downloading or reading the dataset fails.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError(
            "CSV URL must be a non-empty string."
        )

    url = url.strip()

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(
            "Filename must be a non-empty string."
        )

    filename = filename.strip()

    data_dir = Path(data_dir)

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = data_dir / filename

    if dataset_path.exists() and not force_download:

        print(
            f"Using existing dataset:\n"
            f"{dataset_path}"
        )

    else:

        print(
            f"Downloading dataset from:\n"
            f"{url}"
        )

        try:

            response = requests.get(
                url,
                timeout=timeout,
            )

            response.raise_for_status()

            dataset_path.write_bytes(
                response.content
            )

            print(
                f"Dataset downloaded successfully:\n"
                f"{dataset_path}"
            )

        except requests.RequestException as exc:

            if dataset_path.exists():
                dataset_path.unlink()

            raise RuntimeError(
                f"Failed to download dataset from:\n"
                f"{url}"
            ) from exc

    try:

        df = pd.read_csv(
            dataset_path
        )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to read CSV file:\n"
            f"{dataset_path}"
        ) from exc

    if df.empty:

        raise ValueError(
            f"Dataset is empty:\n"
            f"{dataset_path}"
        )

    print(
        f"Dataset loaded successfully:\n"
        f"- Path: {dataset_path}\n"
        f"- Rows: {df.shape[0]:,}\n"
        f"- Columns: {df.shape[1]:,}"
    )

    return df


def load_csv_with_numpy(
    url: str,
    data_dir: str | Path,
    filename: str = "dataset.csv",
    timeout: int = 30,
    force_download: bool = False,
    delimiter: str = ",",
    skip_header: int = 0,
) -> np.ndarray:
    """
    Download and load a CSV dataset using numpy.genfromtxt().

    The dataset is cached locally. If the dataset already exists
    and force_download=False, the existing file will be reused.

    Args:
        url:
            Direct URL to the CSV file.

        data_dir:
            Directory where the dataset will be stored.

        filename:
            Local filename for the dataset.

        timeout:
            HTTP request timeout in seconds.

        force_download:
            If True, download the dataset again even if the
            local file already exists.

        delimiter:
            Character used to separate values in the CSV file.

        skip_header:
            Number of lines to skip at the beginning of the file.

    Returns:
        numpy.ndarray:
            Loaded dataset.

    Raises:
        ValueError:
            If the URL is invalid or the dataset is empty.

        RuntimeError:
            If downloading or reading the dataset fails.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError(
            "CSV URL must be a non-empty string."
        )

    url = url.strip()

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(
            "Filename must be a non-empty string."
        )

    filename = filename.strip()

    data_dir = Path(data_dir)

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = data_dir / filename

    if dataset_path.exists() and not force_download:

        print(
            f"Using existing dataset:\n"
            f"{dataset_path}"
        )

    else:

        print(
            f"Downloading dataset from:\n"
            f"{url}"
        )

        try:

            response = requests.get(
                url,
                timeout=timeout,
            )

            response.raise_for_status()

            dataset_path.write_bytes(
                response.content
            )

            print(
                f"Dataset downloaded successfully:\n"
                f"{dataset_path}"
            )

        except requests.RequestException as exc:

            if dataset_path.exists():
                dataset_path.unlink()

            raise RuntimeError(
                f"Failed to download dataset from:\n"
                f"{url}"
            ) from exc

    try:

        data = np.genfromtxt(
            dataset_path,
            delimiter=delimiter,
            skip_header=skip_header,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to read CSV file with NumPy:\n"
            f"{dataset_path}"
        ) from exc

    if data.size == 0:

        raise ValueError(
            f"Dataset is empty:\n"
            f"{dataset_path}"
        )

    if data.ndim == 1:
        rows = data.shape[0]
        columns = 1
    else:
        rows, columns = data.shape

    print(
        f"Dataset loaded successfully:\n"
        f"- Path: {dataset_path}\n"
        f"- Rows: {rows:,}\n"
        f"- Columns: {columns:,}"
    )

    return data
