"""Create a LAMA.lnk shortcut with the divine orb icon."""
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
print(f"Created: {lnk}")
print(f"  Target: {target}")
print(f"  Icon: {ico}")
