; LAMA (Live Auction Market Assessor) — Inno Setup installer script
; Reads version from resources\VERSION, bundles dist\LAMA\

#define AppName "LAMA"
#define AppPublisher "Couloir"
#define AppURL "https://github.com/CouloirGG/lama"
#define AppExeName "LAMA.exe"

; Read version from resources\VERSION (relative to this .iss file's parent)
#define VersionFile AddBackslash(SourcePath) + "..\resources\VERSION"
#define AppVersion Trim(ReadIni(VersionFile, "", "", "1.0.0"))
; ReadIni won't work for plain text — use FileOpen instead
#undef AppVersion
#define FileHandle FileOpen(VersionFile)
#if FileHandle
  #define AppVersion Trim(FileRead(FileHandle))
  #expr FileClose(FileHandle)
#else
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{8F3E4A72-B1D9-4C5E-A8F2-1D3E5B7C9A04}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} v{#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={localappdata}\LAMA
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=LAMA-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=lowest
CloseApplications=yes
RestartApplications=no
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Bundle the entire PyInstaller output directory
Source: "..\dist\LAMA\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\LAMA\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall
