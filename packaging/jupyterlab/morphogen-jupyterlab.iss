; The Windows installer of the JupyterLab environment of morphoGen (Inno Setup 6): the folder
; that build_environment.py makes (a Python of its own, morphoGen, JupyterLab), installed for
; the current user without administrator rights, with a shortcut in the Start menu that opens
; JupyterLab on the user's notebooks, and an uninstaller. From the root of the repository,
; once the environment is built:
;
;     iscc /DAppVersion=1.0.2 packaging\jupyterlab\morphogen-jupyterlab.iss
;
; gives dist\morphoGen-JupyterLab-1.0.2-windows-x64-setup.exe. See packaging/jupyterlab/README.md.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Environment
  #define Environment "..\..\build\jupyterlab"
#endif

[Setup]
; the same identity in every version: a new version replaces the installed one
AppId={{CA5891F7-4A6F-4217-A6C7-34AA214EBFAE}
AppName=morphoGen JupyterLab
AppVersion={#AppVersion}
AppVerName=morphoGen JupyterLab {#AppVersion}
AppPublisher=Justin Dirrenberger
AppPublisherURL=https://github.com/jdirr/morphoGen
AppSupportURL=https://github.com/jdirr/morphoGen/issues
AppUpdatesURL=https://github.com/jdirr/morphoGen/releases
AppCopyright=Copyright (C) 2026 Justin Dirrenberger
; for the current user (%LOCALAPPDATA%\Programs) unless an administrator chooses all users
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline dialog
DefaultDirName={autopf}\morphoGen JupyterLab
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\..\dist
OutputBaseFilename=morphoGen-JupyterLab-{#AppVersion}-windows-x64-setup
SetupIconFile=..\..\src\morphogen\gui\icon.ico
UninstallDisplayIcon={app}\morphogen.ico
UninstallDisplayName=morphoGen JupyterLab {#AppVersion}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
; a running JupyterLab of this environment is closed (after asking) before its files are replaced
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[CustomMessages]
english.Comment=JupyterLab with morphoGen, on the notebooks of Documents\morphoGen
french.Comment=JupyterLab avec morphoGen, sur les carnets de Documents\morphoGen

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; a new version replaces the environment instead of mixing two sets of packages
Type: filesandordirs; Name: "{app}\python"

[Files]
Source: "{#Environment}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\src\morphogen\gui\icon.ico"; DestDir: "{app}"; DestName: "morphogen.ico"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\morphoGen JupyterLab"; Filename: "{app}\python\python.exe"; Parameters: """{app}\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\morphogen.ico"; Comment: "{cm:Comment}"
Name: "{autodesktop}\morphoGen JupyterLab"; Filename: "{app}\python\python.exe"; Parameters: """{app}\launcher.py"""; WorkingDir: "{app}"; IconFilename: "{app}\morphogen.ico"; Comment: "{cm:Comment}"; Tasks: desktopicon

[Run]
Filename: "{app}\python\python.exe"; Parameters: """{app}\launcher.py"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,morphoGen JupyterLab}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; what Python and Numba write beside the packages (their caches); the notebooks of the
; user's documents stay
Type: filesandordirs; Name: "{app}\python"
