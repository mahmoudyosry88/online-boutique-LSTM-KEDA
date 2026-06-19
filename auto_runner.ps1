$ErrorActionPreference = "Stop"

function Run-Cmd {
    param([string]$cmd)
    Write-Host "
>>> Executing: $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
}

function Reset-Cluster {
    Write-Host "
--- RESETTING CLUSTER ---" -ForegroundColor Yellow
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.Id -ne $PID} | Stop-Process -Force
    Run-Cmd "kubectl delete hpa --all -n default"
    Run-Cmd "kubectl delete scaledobject --all -n default"
    Run-Cmd "kubectl scale deployment -n default --replicas=1 --all"
    Run-Cmd "kubectl patch deployment loadgenerator -n default --patch-file c:\ex1\microservices-demo\loadgen-patch-40.yaml"
    Start-Sleep -Seconds 10
}

function Run-Phase {
    param(
        [string]$PhaseName,
        [string]$OutputCsv,
        [string]$LogJson
    )
    Write-Host "
=== STARTING PHASE: $PhaseName ===" -ForegroundColor Green
    
    $baseUtc = (Get-Date).ToUniversalTime()
    
    $segments = @(
        @{ duration = 5; users = 40; patch = "loadgen-patch-40.yaml" },
        @{ duration = 10; users = 800; patch = "loadgen-patch-800.yaml" },
        @{ duration = 7; users = 40; patch = "loadgen-patch-40.yaml" }
    )
    
    $timeline = @()
    $currentTime = $baseUtc
    
    foreach ($seg in $segments) {
        Write-Host "
[16:52:30] Applying patch for $($seg.users) users..." -ForegroundColor Magenta
        Run-Cmd "kubectl patch deployment loadgenerator -n default --patch-file c:\ex1\microservices-demo\$($seg.patch)"
        
        $endTime = $currentTime.AddMinutes($seg.duration)
        
        $timeline += @{
            start_time = $currentTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            end_time   = $endTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
            users      = $seg.users
        }
        
        Write-Host "Waiting $($seg.duration) minutes..."
        Start-Sleep -Seconds ($seg.duration * 60)
        $currentTime = $endTime
    }
    
    Write-Host "
Writing user log JSON..."
    $timeline | ConvertTo-Json -Depth 10 | Out-File -FilePath $LogJson -Encoding UTF8
    
    Write-Host "
Configuring collection script..."
    $collectScript = "c:\ex1\microservices-demo\lstm_autoscaler\scripts\live_production_daemon\collect_metrics_live.py"
    $code = Get-Content $collectScript -Raw
    $code = $code -replace "OUTPUT_FILE = .*", "OUTPUT_FILE = r'$OutputCsv'"
    $code = $code -replace "USERS_LOG_FILE = .*", "USERS_LOG_FILE = r'$LogJson'"
    $code = $code -replace "HOURS\s*=.*", "HOURS = 0.4"
    Set-Content -Path $collectScript -Value $code -Encoding UTF8
    
    Write-Host "
Running metrics collection..."
    Run-Cmd "python $collectScript"
}

Reset-Cluster
Write-Host "
Applying HPA Baseline..."
Run-Cmd "kubectl apply -f c:\ex1\microservices-demo\kubernetes-manifests\hpa_all_services.yaml"
Start-Sleep -Seconds 10

$outCsv = "c:\ex1\microservices-demo\lstm_autoscaler\outputs\live_comparison_results\hpa_live_dataset.csv"
$outJson = "c:\ex1\microservices-demo\lstm_autoscaler\outputs\live_comparison_results\hpa_users_log.json"
Run-Phase -PhaseName "HPA Baseline" -OutputCsv $outCsv -LogJson $outJson

Write-Host "
--- Phase 1 Complete. Waiting 5 minutes for stabilization ---" -ForegroundColor Yellow
Start-Sleep -Seconds 300

Reset-Cluster
Write-Host "
Applying KEDA ScaledObjects..."
Run-Cmd "kubectl apply -f c:\ex1\microservices-demo\kubernetes-manifests\scaled_objects_all_services.yaml"

Write-Host "
Starting LSTM Daemon in background..."
Start-Process -FilePath "python" -ArgumentList "-u live_predictor.py" -WorkingDirectory "c:\ex1\microservices-demo\lstm_autoscaler\scripts\live_production_daemon" -RedirectStandardOutput "lstm_keda_debug.log" -RedirectStandardError "lstm_keda_debug.log" -NoNewWindow
Start-Sleep -Seconds 30

$outCsv2 = "c:\ex1\microservices-demo\lstm_autoscaler\outputs\live_comparison_results\lstm_live_dataset.csv"
$outJson2 = "c:\ex1\microservices-demo\lstm_autoscaler\outputs\live_comparison_results\lstm_users_log.json"
Run-Phase -PhaseName "LSTM+KEDA Proactive" -OutputCsv $outCsv2 -LogJson $outJson2

Write-Host "
Stopping LSTM Daemon..."
Get-Process -Name "python" -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "
--- Generating Plots ---"
Run-Cmd "python plot_all_services_twinx.py"
Write-Host "
=== EXPERIMENT FINISHED SUCCESSFULLY ===" -ForegroundColor Green
