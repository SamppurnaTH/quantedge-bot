# Schedule Trading Bot Reports for Windows
# This script creates two scheduled tasks:
# 1. Pre-Market Scan at 8:45 AM IST
# 2. Post-Market Outlook at 4:00 PM IST

$Action = New-ScheduledTaskAction -Execute "python.exe" -Argument "b:\Personal\Bot\trading_bot\run_daily.py" -WorkingDirectory "b:\Personal\Bot\trading_bot"

# Task 1: 8:45 AM
$Trigger1 = New-ScheduledTaskTrigger -Daily -At 8:45am
Register-ScheduledTask -Action $Action -Trigger $Trigger1 -TaskName "QuantEdge_PreMarket" -Description "Daily Pre-Market Trading Signal Scan" -Force

# Task 2: 4:00 PM
$Trigger2 = New-ScheduledTaskTrigger -Daily -At 4:00pm
Register-ScheduledTask -Action $Action -Trigger $Trigger2 -TaskName "QuantEdge_PostMarket" -Description "Daily Post-Market Outlook and Reporting" -Force

Write-Host "✅ Tasks scheduled successfully!" -ForegroundColor Green
Write-Host "You will now receive Telegram reports at 8:45 AM and 4:00 PM daily."
