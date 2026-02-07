"""Restore .scan_complete files that were accidentally deleted"""

from pathlib import Path

parent_folder = Path(r"\\129.22.141.165\Data\softbox_scan\P0_new\20251020_Quel_new")

restored_count = 0

for child_folder in parent_folder.iterdir():
    if not child_folder.is_dir():
        continue

    # Check if it's a scan folder (starts with "scan_")
    if not child_folder.name.startswith("scan_"):
        continue

    # Check if .scan_complete exists
    scan_complete = child_folder / ".scan_complete"

    if not scan_complete.exists():
        # Create the file
        scan_complete.touch()
        print(f"Restored .scan_complete in {child_folder.name}")
        restored_count += 1
    else:
        print(f"Already exists in {child_folder.name}")

print(f"\nRestored {restored_count} .scan_complete files")
