"""
Automatyczna instalacja Hadoop binaries dla Spark na Windows
Zawiera winutils.exe + hadoop.dll (native library)
"""

import os
import urllib.request
import sys


def download_hadoop_binaries():
    """Pobierz i zainstaluj pełne Hadoop binaries dla Windows"""

    print("="*60)
    print("SETUP HADOOP BINARIES DLA SPARK/WINDOWS")
    print("="*60)

    # Ścieżki
    project_dir = os.path.dirname(os.path.abspath(__file__))
    hadoop_dir = os.path.join(project_dir, 'hadoop')
    bin_dir = os.path.join(hadoop_dir, 'bin')

    # Utwórz katalogi
    os.makedirs(bin_dir, exist_ok=True)

    # Pliki do pobrania (Hadoop 3.3.5 - kompatybilny z Spark 3.5)
    files_to_download = {
        'winutils.exe': 'https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/winutils.exe',
        'hadoop.dll': 'https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.5/bin/hadoop.dll',
    }

    downloaded = []

    for filename, url in files_to_download.items():
        file_path = os.path.join(bin_dir, filename)

        # Sprawdz czy juz istnieje
        if os.path.exists(file_path):
            print(f"[OK] {filename} already exists")
            print(f"     Size: {os.path.getsize(file_path):,} bytes")
            downloaded.append(filename)
            continue

        print(f"\nPobieranie {filename}...")
        print(f"  URL: {url}")

        try:
            urllib.request.urlretrieve(url, file_path)
            size = os.path.getsize(file_path)
            print(f"  [OK] Pobrano! Rozmiar: {size:,} bytes")
            downloaded.append(filename)

        except Exception as e:
            print(f"  [ERROR] Błąd podczas pobierania: {e}")
            return False

    # Ustaw zmienne środowiskowe
    os.environ['HADOOP_HOME'] = hadoop_dir
    os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')

    print("\n" + "="*60)
    print("INSTALACJA ZAKOŃCZONA!")
    print("="*60)
    print(f"\nHADOOP_HOME: {hadoop_dir}")
    print(f"Pobrane pliki:")
    for f in downloaded:
        print(f"  ✓ {f}")

    print("\n⚠️  WAŻNE:")
    print("  Jeśli masz otwartą sesję Spark, ZRESTARTUJ ją!")
    print("  Restart Jupyter kernel: Kernel → Restart")

    return True


if __name__ == "__main__":
    success = download_hadoop_binaries()

    if not success:
        print("\n⚠️  INSTALACJA NIEUDANA")
        print("\nRozwiązanie alternatywne:")
        print("1. Pobierz ręcznie z: https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin")
        print("2. Skopiuj winutils.exe i hadoop.dll do: hadoop/bin/")
        sys.exit(1)

    print("\n" + "="*60)
    print("✅ GOTOWE!")
    print("="*60)
    print("\nKolejne kroki:")
    print("1. ZRESTARTUJ Jupyter kernel")
    print("2. Uruchom ponownie notebook od początku")
    print("3. Spark powinien działać poprawnie!")
    print("="*60)
