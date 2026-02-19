"""Create a LAMA.lnk shortcut with the divine orb icon and AppUserModelID."""
import os
import sys
import subprocess

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_dir, "src"))
from bundle_paths import get_resource

ico = str(get_resource("resources/img/favicon.ico")).replace("/", "\\")
target = os.path.join(project_dir, "LAMA.pyw").replace("/", "\\")
lnk = os.path.join(project_dir, "LAMA.lnk").replace("/", "\\")
working_dir = project_dir.replace("/", "\\")

# Step 1: Create the shortcut with WScript.Shell
ps_script = (
    '$ws = New-Object -ComObject WScript.Shell; '
    '$s = $ws.CreateShortcut("' + lnk + '"); '
    '$s.TargetPath = "' + target + '"; '
    '$s.WorkingDirectory = "' + working_dir + '"; '
    '$s.IconLocation = "' + ico + ',0"; '
    '$s.Description = "LAMA - Live Auction Market Assessor"; '
    '$s.Save()'
)
subprocess.run(["powershell", "-Command", ps_script], check=True)

# Step 2: Stamp AppUserModelID so Windows matches pinned shortcut to running app
try:
    from win32com.propsys import propsys
    store = propsys.SHGetPropertyStoreFromParsingName(
        lnk, None, 2  # GPS_READWRITE
    )
    key_id = propsys.PSGetPropertyKeyFromName("System.AppUserModel.ID")
    store.SetValue(key_id, propsys.PROPVARIANTType("Couloir.LAMA"))
    key_icon = propsys.PSGetPropertyKeyFromName(
        "System.AppUserModel.RelaunchIconResource"
    )
    store.SetValue(key_icon, propsys.PROPVARIANTType(ico))
    store.Commit()
    print("  AppUserModelID: Couloir.LAMA (set via IPropertyStore)")
except Exception as e:
    print(f"  Warning: could not set AppUserModelID: {e}")

print(f"Created: {lnk}")
print(f"  Target: {target}")
print(f"  Icon: {ico}")
