import os
import wget
import sys

# URL for SAM2 Tiny Checkpoint
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
FILENAME = "sam2.1_hiera_tiny.pt"

def download_sam2():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(base_dir, FILENAME)

        if os.path.exists(target_path):
            print(f"File already exists at: {target_path}")
            print(f"Size: {os.path.getsize(target_path) / (1024*1024):.2f} MB")
            choice = input("Do you want to re-download? (y/n): ")
            if choice.lower() != 'y':
                return

        print(f"Downloading {FILENAME} from {SAM2_URL}...")
        wget.download(SAM2_URL, target_path)
        print(f"\nSuccessfully downloaded to {target_path}")

    except ImportError:
        print("Error: 'wget' library is missing. Installing it now...")
        os.system(f"{sys.executable} -m pip install wget")
        import wget
        download_sam2()
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_sam2()
    input("\nPress Enter to exit...")
