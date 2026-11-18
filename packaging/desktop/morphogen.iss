; The Windows installer of the stand-alone application of morphoGen (Inno Setup 6): the
; folder that PyInstaller makes (morphogen-desktop.spec), installed for the current user
; without administrator rights, with a shortcut in the Start menu and an uninstaller. From
; the root of the repository, once the bundle is built:
;
;     iscc /DAppVersion=1.0.0 packaging\desktop\morphogen.iss
;
; gives dist\morphoGen-1.0.0-setup.exe. See packaging/desktop/README.md.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Bundle
  #define Bundle "..\..\dist\morphoGen"
#endif

[Setup]
; the same identity in every version: a new version replaces the installed one
AppId={{A466015B-5487-4768-874A-77F8CDE59D3F}
AppName=morphoGen
AppVersion={#AppVersion}
AppVerName=morphoGen {#AppVersion}
AppPublisher=Justin Dirrenberger
AppPublisherURL=https://github.com/jdirr/morphoGen
AppSupportURL=https://github.com/jdirr/morphoGen/issues
AppUpdatesURL=https://github.com/jdirr/morphoGen/releases
AppCopyright=Copyright (C) 2026 Justin Dirrenberger
; for the current user (%LOCALAPPDATA%\Programs) unless an administrator chooses all users
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline dialog
DefaultDirName={autopf}\morphoGen
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\..\dist
OutputBaseFilename=morphoGen-{#AppVersion}-setup
SetupIconFile=..\..\src\morphogen\gui\icon.ico
UninstallDisplayIcon={app}\morphoGen.exe
UninstallDisplayName=morphoGen {#AppVersion}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
; a running morphoGen is closed (after asking) before its files are replaced
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[CustomMessages]
english.Comment=Virtual microstructures for the mechanics and physics of materials
french.Comment=Microstructures virtuelles pour la mécanique et la physique des matériaux
english.NoWebView2=morphoGen shows its window with Microsoft Edge WebView2, which this computer does not seem to have: install its Evergreen runtime (https://developer.microsoft.com/microsoft-edge/webview2/) before starting morphoGen.%n%nContinue the installation?
french.NoWebView2=morphoGen affiche sa fenêtre avec Microsoft Edge WebView2, qui ne semble pas installé sur cet ordinateur : installez son runtime Evergreen (https://developer.microsoft.com/microsoft-edge/webview2/) avant de lancer morphoGen.%n%nPoursuivre l'installation ?

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#Bundle}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\morphoGen"; Filename: "{app}\morphoGen.exe"; Comment: "{cm:Comment}"
Name: "{autodesktop}\morphoGen"; Filename: "{app}\morphoGen.exe"; Comment: "{cm:Comment}"; Tasks: desktopicon

[Run]
Filename: "{app}\morphoGen.exe"; Description: "{cm:LaunchProgram,morphoGen}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; what the application writes beside its own files (the caches of Python and Numba)
Type: filesandordirs; Name: "{app}\_internal"

[Code]
const
  WebView2Client = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

function HasWebView2: Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + WebView2Client, 'pv', Version)
    or RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\' + WebView2Client, 'pv', Version);
  Result := Result and (Version <> '') and (Version <> '0.0.0.0');
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  if not HasWebView2 then
    Result := SuppressibleMsgBox(CustomMessage('NoWebView2'), mbConfirmation, MB_YESNO, IDYES) = IDYES;
end;
