Start-Sleep -Seconds 12
Write-Host "=== BACKEND STDERR ==="
Get-Content 'C:\Users\RX535PT\Documents\rec\backend_err.log' -ErrorAction SilentlyContinue
Write-Host "=== BACKEND STDOUT ==="
Get-Content 'C:\Users\RX535PT\Documents\rec\backend.log' -ErrorAction SilentlyContinue
