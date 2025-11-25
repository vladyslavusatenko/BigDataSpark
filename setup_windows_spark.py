"""
Automatyczna instalacja winutils.exe dla Spark na Windows
"""

import os
import urllib.request
import sys


def download_winutils():
    """Pobierz i zainstaluj winutils.exe dla Windows"""

    print("="*60)
    print("SETUP SPARK DLA WINDOWS")
    print("="*60)

    # Ścieżki
    project_dir = os.path.dirname(os.path.abspath(__file__))
    hadoop_dir = os.path.join(project_dir, 'hadoop')
    bin_dir = os.path.join(hadoop_dir, 'bin')
    winutils_path = os.path.join(bin_dir, 'winutils.exe')

    # Utwórz katalogi
    os.makedirs(bin_dir, exist_ok=True)

    # Sprawdź czy już istnieje
    if os.path.exists(winutils_path):
        print(f"[OK] winutils.exe juz istnieje: {winutils_path}")
        print(f"  Rozmiar: {os.path.getsize(winutils_path)} bytes")
        return True

    # URL do winutils.exe dla Hadoop 3.3.6 (kompatybilny z Spark 3.5)
    winutils_url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe"

    print(f"\nPobieranie winutils.exe z GitHub...")
    print(f"URL: {winutils_url}")
    print(f"Cel: {winutils_path}")

    try:
        # Pobierz plik
        print("Downloading...")
        urllib.request.urlretrieve(winutils_url, winutils_path)

        print(f"\n[OK] winutils.exe pobrane pomyslnie!")
        print(f"  Lokalizacja: {winutils_path}")
        print(f"  Rozmiar: {os.path.getsize(winutils_path)} bytes")

        # Ustaw zmienne środowiskowe
        os.environ['HADOOP_HOME'] = hadoop_dir
        print(f"\n[OK] HADOOP_HOME ustawione: {hadoop_dir}")

        print("\n" + "="*60)
        print("INSTALACJA ZAKONCZONA POMYSLNIE!")
        print("="*60)
        print("\nMozesz teraz uruchomic notebook EDA.")
        print("Zmienne srodowiskowe zostana automatycznie ustawione przez SparkConfig.")

        return True

    except Exception as e:
        print(f"\n[ERROR] Blad podczas pobierania: {e}")
        print("\nAlternatywne rozwiązanie:")
        print("1. Pobierz ręcznie z: https://github.com/steveloughran/winutils")
        print(f"2. Umieść w: {winutils_path}")
        return False


if __name__ == "__main__":
    success = download_winutils()

    if not success:
        sys.exit(1)

    print("\n" + "="*60)
    print("GOTOWE! Uruchom ponownie notebook.")
    print("="*60)
