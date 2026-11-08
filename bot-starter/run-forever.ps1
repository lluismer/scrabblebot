# Keep the bot running: restart it whenever it exits (a dropped connection
# makes it exit on purpose). Stop with Ctrl+C twice.
#
#   cd bot-starter
#   powershell -ExecutionPolicy Bypass -File run-forever.ps1 -Name sepcis
#   # entering a tournament:
#   powershell -ExecutionPolicy Bypass -File run-forever.ps1 -Name sepcis -Tournament 2
#   # the 7-letter rush brain (src/rush/rush.ts) instead of strategy C,
#   # optionally with overrides:
#   powershell -ExecutionPolicy Bypass -File run-forever.ps1 -Name sepcis -Brain rush
#   powershell -ExecutionPolicy Bypass -File run-forever.ps1 -Name sepcis -Brain rush -RushParams "picky=8,lateBag=20"

param(
  [string]$Name = "sepcis",
  [string]$Tournament = "",
  [string]$Brain = "",
  [string]$RushParams = ""
)

$env:BOT_NAME = $Name
if ($Tournament -ne "") { $env:TOURNAMENT_ID = $Tournament }
$env:BRAIN = $Brain
$env:RUSH_PARAMS = $RushParams
Set-Location $PSScriptRoot

while ($true) {
  npm start
  Write-Host "[$Name] bot exited (code $LASTEXITCODE); restarting in 3 s"
  Start-Sleep -Seconds 3
}
