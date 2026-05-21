# Schedule Trading Bot Reports for Windows
# This script creates two scheduled tasks:
# 1. Pre-Market Scan at 8:30 AM local time
# 2. End-Of-Day Outlook at 3:15 PM local time
# Note: The bot now logs both local clock and IST timestamps to help detect timezone mismatch.

$pythonPath = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $pythonPath) {
    Write-Host "ERROR: python.exe was not found on PATH. Please install Python or update the script to use the full interpreter path." -ForegroundColor Red
    exit 1
}

$Action = New-ScheduledTaskAction -Execute $pythonPath -Argument "b:\Personal\Bot\trading_bot\run_daily.py" -WorkingDirectory "b:\Personal\Bot\trading_bot"

# Task 1: 8:30 AM
$Trigger1 = New-ScheduledTaskTrigger -Daily -At 8:30am
Register-ScheduledTask -Action $Action -Trigger $Trigger1 -TaskName "QuantEdge_PreMarket" -Description "Daily Pre-Market Trading Signal Scan" -Force

# Task 2: 3:15 PM
$Trigger2 = New-ScheduledTaskTrigger -Daily -At 3:15pm
Register-ScheduledTask -Action $Action -Trigger $Trigger2 -TaskName "QuantEdge_PostMarket" -Description "Daily Post-Market Outlook and Reporting" -Force

Write-Host "✅ Tasks scheduled successfully!" -ForegroundColor Green
Write-Host "Pre-Market scan scheduled at 08:30 and EOD audit scheduled at 15:15 (local time)."
Write-Host "If your machine is not in IST, adjust scheduling or use a timezone-aware scheduler instead."
