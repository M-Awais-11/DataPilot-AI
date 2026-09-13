@echo off
:: ============================================================
:: SQL Server Fix Script - Self-elevating to Administrator
:: ============================================================
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo ============================================================
echo  SQL Server Fix - Running as Administrator
echo ============================================================
echo.

echo [1/5] Enabling TCP/IP for MSSQL17.SQLEXPRESS...
reg add "HKLM\SOFTWARE\Microsoft\Microsoft SQL Server\MSSQL17.SQLEXPRESS\MSSQLServer\SuperSocketNetLib\Tcp" /v Enabled /t REG_DWORD /d 1 /f
reg add "HKLM\SOFTWARE\Microsoft\Microsoft SQL Server\MSSQL17.SQLEXPRESS\MSSQLServer\SuperSocketNetLib\Tcp\IPAll" /v TcpPort /t REG_SZ /d 1433 /f
reg add "HKLM\SOFTWARE\Microsoft\Microsoft SQL Server\MSSQL17.SQLEXPRESS\MSSQLServer\SuperSocketNetLib\Tcp\IPAll" /v TcpDynamicPorts /t REG_SZ /d "" /f
echo     TCP/IP enabled on port 1433.

echo [2/5] Enabling and starting SQL Server Browser...
sc config SQLBrowser start= auto
sc start SQLBrowser
echo     SQL Browser started.

echo [3/5] Adding Windows Firewall rules...
netsh advfirewall firewall delete rule name="SQL Server TCP 1433" >nul 2>&1
netsh advfirewall firewall delete rule name="SQL Browser UDP 1434" >nul 2>&1
netsh advfirewall firewall add rule name="SQL Server TCP 1433" protocol=TCP dir=in localport=1433 action=allow
netsh advfirewall firewall add rule name="SQL Browser UDP 1434" protocol=UDP dir=in localport=1434 action=allow
echo     Firewall rules added.

echo [4/5] Restarting MSSQL$SQLEXPRESS...
net stop "MSSQL$SQLEXPRESS" /y
timeout /t 3 /nobreak >nul
net start "MSSQL$SQLEXPRESS"
echo     SQL Server restarted.

echo [5/5] Verifying port 1433...
timeout /t 5 /nobreak >nul
powershell -Command "if ((Test-NetConnection -ComputerName localhost -Port 1433).TcpTestSucceeded) { Write-Host '    PORT 1433: OPEN - Fix successful!' } else { Write-Host '    PORT 1433: still closed - check SQL Server logs.' }"

echo.
echo ============================================================
echo  Done! Restart the backend now.
echo ============================================================
pause
