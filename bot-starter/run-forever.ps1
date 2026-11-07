# Keep the bot running: restart it whenever it exits (a dropped connection
# makes it exit on purpose). Stop with Ctrl+C twice.
#
#   cd bot-starter
#   powershell -ExecutionPolicy Bypass -File run-forever.ps1 -Name sepcis
#   # entering a tournament:
#   powershell -ExecutionPolicy Bypass -File run-forever.ps1 -Name sepcis -Tournament 2

param(
  [string]$Name = "sepcis",
  [string]$Tournament = ""
)

$env:BOT_NAME = $Name
if ($Tournament -ne "") { $env:TOURNAMENT_ID = $Tournament }
Set-Location $PSScriptRoot

while ($true) {
  npm start
  Write-Host "[$Name] bot exited (code $LASTEXITCODE); restarting in 3 s"
  Start-Sleep -Seconds 3
}
