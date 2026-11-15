param(
    [Parameter(Mandatory = $true)][string]$InputDirectory,
    [string]$Pattern = "*.cdxml"
)

$ErrorActionPreference = "Stop"
$files = @(Get-ChildItem -LiteralPath $InputDirectory -Filter $Pattern -File)
if ($files.Count -eq 0) { throw "No ChemDraw files matched the requested input." }
$app = New-Object -ComObject ChemDraw.Application
$initialDocuments = $app.Documents.Count
$doc = $null
try {
    foreach ($file in $files) {
        $doc = $app.Documents.Open($file.FullName)
        $output = [IO.Path]::ChangeExtension($file.FullName, ".svg")
        $doc.SaveAs($output)
        if (-not (Test-Path -LiteralPath $output)) { throw "SVG export missing: $output" }
        $doc.Close($false)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc)
        $doc = $null
        Write-Output $output
    }
}
finally {
    if ($null -ne $doc) {
        $doc.Close($false)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc)
    }
    if ($initialDocuments -eq 0 -and $app.Documents.Count -eq 0) { $app.Quit() }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
}
