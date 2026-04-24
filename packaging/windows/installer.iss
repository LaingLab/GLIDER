; Inno Setup script for the GLIDER Windows installer.
;
; Builds a per-user installer (no UAC prompt) that writes to
; %LocalAppData%\Programs\GLIDER, adds Start Menu and optional desktop
; shortcuts, and registers a clean uninstaller.
;
; Build with:
;     ISCC.exe packaging\windows\installer.iss
; Output: Output\glider-setup-<version>.exe

#define MyAppName      "GLIDER"
#define MyAppPublisher "LaingLab"
#define MyAppURL       "https://github.com/LaingLab/glider"
#define MyAppExeName   "GLIDER.exe"

; The version string is substituted by CI from src/glider/_version.py. Locally
; you can pass it on the command line: ISCC /DMyAppVersion=1.2.3 installer.iss
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif

[Setup]
; A stable GUID so upgrades replace the previous install in place.
; Generated once for this app — do not change this value across releases.
AppId={{7B6C8F2E-3D1A-4C5B-9E0F-A8F7D3C6B1E4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=glider-setup-{#MyAppVersion}
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; Per-user install avoids a UAC prompt — lab staff log in as themselves and
; don't need admin. A shared-lab variant would flip PrivilegesRequired to
; "admin" and use {commonpf} above. Start with the frictionless path.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UninstallDisplayIcon={app}\{#MyAppExeName}

; Require Windows 10 or later. MinVersion format is <major>.<minor>.<build>.
; 10.0.17763 = Windows 10 1809 (the first LTSC that sees wide lab deployment).
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Package the entire dist/GLIDER folder produced by PyInstaller. Recurse so
; we pick up Qt plugins, OpenCV DLLs, and everything PyInstaller's COLLECT
; step assembled.
Source: "..\..\dist\GLIDER\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; \
    Filename: "{uninstallexe}"

[Run]
Filename: "{app}\{#MyAppExeName}"; \
    Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent
